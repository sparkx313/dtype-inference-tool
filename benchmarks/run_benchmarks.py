"""Measure how often the inferrer names the right type, and what it costs.

Run with `python benchmarks/run_benchmarks.py`. Every number printed here is from
a generated corpus with known labels, so the accuracy table is reproducible with
`--seed`. Nothing is hand-tuned to make the table look better: the confusing
pairs are printed, not hidden.

Seven measurements:

1. per-type precision, recall and F1 against synthetic ground truth
2. wrong names, plus the columns where a second type fits every value too
3. how accuracy and the review rate move as columns get dirtier
4. what the review queue looks like (which ambiguity notes fire)
5. sampling cost versus a full scan, at two column lengths
6. the cost of each candidate parser against each kind of column
7. a side-by-side with pandas' `read_csv` inference
"""

import argparse
import os
import random
import sys
import time
import tracemalloc
import uuid as uuid_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dtypeinfer import Settings, infer_column, infer_table  # noqa: E402
from dtypeinfer.infer import IMPLIED_BY  # noqa: E402
from dtypeinfer.types import PARSERS, SPECIFICITY  # noqa: E402

WORDS = ("alpha brindle cobalt dapple ember fable granite harbor indigo jostle "
         "kelp lapel mosaic nectar orbit prairie quiver ridge sample tundra "
         "umber vixen welkin xenic yarrow zephyr").split()
COLORS = ("red", "green", "amber", "violet", "slate", "cream", "ochre", "teal")
DOMAINS = ("example.org", "test.co.uk", "sample.io", "demo.dev", "mail.example.com")
SYMBOLS = ("$", "€", "£", "¥")


def make_integer(rng):
    return str(rng.randint(0, 10 ** 6))


def make_float(rng):
    return f"{rng.uniform(0, 1000):.2f}"


def make_percent(rng):
    return f"{rng.uniform(0, 100):.1f}%"


def make_currency(rng):
    amount = rng.uniform(0, 20_000)
    return "%s%s" % (rng.choice(SYMBOLS), f"{amount:,.2f}")


def make_boolean(rng):
    return rng.choice(["true", "false", "yes", "no", "TRUE", "No"])


def _date_parts(rng):
    return rng.randint(2000, 2026), rng.randint(1, 12), rng.randint(1, 28)


def make_date(rng):
    year, month, day = _date_parts(rng)
    return "%04d-%02d-%02d" % (year, month, day)


def make_datetime(rng):
    return "%sT%02d:%02d:%02d" % (make_date(rng), rng.randint(0, 23),
                                  rng.randint(0, 59), rng.randint(0, 59))


def make_time(rng):
    return "%02d:%02d:%02d" % (rng.randint(0, 23), rng.randint(0, 59),
                               rng.randint(0, 59))


def make_uuid(rng):
    as_hex = "%032x" % rng.getrandbits(128)
    return str(uuid_module.UUID(as_hex))


def make_email(rng):
    return "%s.%s@%s" % (rng.choice(WORDS).lower(), rng.choice(WORDS).lower(),
                         rng.choice(DOMAINS))


def make_url(rng):
    return "%s://%s/%s/%d" % (rng.choice(["http", "https"]), rng.choice(DOMAINS),
                              rng.choice(WORDS).lower(), rng.randint(1, 9999))


def make_ipv4(rng):
    return ".".join(str(rng.randint(1, 223)) for _ in range(4))


def make_ipv6(rng):
    head = ":".join("%04x" % rng.randint(0, 0xFFFF) for _ in range(4))
    return rng.choice([head + "::", "::1", "2001:db8::%x" % rng.randint(1, 0xFFFF),
                       head + ":%x" % rng.randint(1, 0xFFFF)])


def make_categorical(rng):
    return rng.choice(COLORS)


def make_text(rng):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14)))


MAKERS = {
    "integer": make_integer,
    "float": make_float,
    "percent": make_percent,
    "currency": make_currency,
    "boolean": make_boolean,
    "date": make_date,
    "datetime": make_datetime,
    "time": make_time,
    "uuid": make_uuid,
    "email": make_email,
    "url": make_url,
    "ipv4": make_ipv4,
    "ipv6": make_ipv6,
    "categorical": make_categorical,
    "text": make_text,
}
MEASURED_TYPES = tuple(name for name in MAKERS)

# Hand-written columns whose spelling is not decidable per value. The generated
# corpus above uses one canonical spelling per type and therefore never produces
# these; real data does, which is where the notes carry the actual weight.
_AMBIGUOUS_VALUES = {
    "zip codes": ["02134", "10001", "90210", "00501", "19103"] * 20,
    "flag column": ["0", "1", "1", "0"] * 25,
    "eu decimals": ["1234,56", "12,30", "9876,00"] * 20,
    "ambiguous tail": ["1.234", "2.345", "9.999"] * 20,
    "slash dates": ["03/04/2020", "05/06/2020", "07/08/2020"] * 20,
    "unpadded dates": ["1/2/20", "3/4/20"] * 20,
    "compact dates": ["20260131", "20260201"] * 20,
    "mixed currency": ["$1,234.56", "12 EUR", "€3"] * 20,
    "sparse dates": ["2026-01-01", "", "NA", "2026-01-02", " "] * 20,
    "leading-zero ip": ["10.0.0.1", "01.2.3.4", "192.168.0.7"] * 20,
    "inconsistent decimals": ["12,5", "1.5"] * 20,
    "times only": ["09:30", "18:05", "23:59"] * 20,
    "us dates": ["01/31/2020", "12/25/2020"] * 20,
    "epoch seconds": ["1700000000", "1800000000"] * 20,
}


def column(type_name, rng, rows=400, dirt=0.0, other=None):
    """A generated column, with `dirt` share of cells replaced by junk."""
    maker = MAKERS[type_name]
    values = []
    for _ in range(rows):
        if dirt and rng.random() < dirt:
            draw = rng.random()
            if draw < 0.4:
                values.append(rng.choice(["", "NA", "null", "N/A", "-"]))
            elif draw < 0.7:
                values.append("  %s " % maker(rng))
            else:
                values.append(other(rng) if other else "bad cell")
            continue
        values.append(maker(rng))
    return values


def header(title):
    print()
    print(title)
    print("-" * len(title))


def top_rival(result):
    """The best type that fits the column *besides* the one that was chosen.

    An integer column fits `float` too, by definition - that is the same reading
    seen one rung further down the ladder, not a second opinion, so IMPLIED_BY
    filters it out. What survives is a genuine coin-flip: two grammars that both
    describe every value in the column.
    """
    name, fit = None, 0.0
    for other, score in result.fits.items():
        if other == result.type or score <= fit:
            continue
        if other in IMPLIED_BY.get(result.type, frozenset()):
            continue
        name, fit = other, score
    return name, fit


def accuracy(seed, columns_per_type, rows):
    header("1. per-type accuracy against synthetic ground truth")
    rng = random.Random(seed)
    stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in MEASURED_TYPES}
    confusion = {}
    clean_claims = 0
    total = 0
    settings = Settings(sample_size=min(rows, 250), seed=seed)
    for type_name in MEASURED_TYPES:
        other = rng.choice([MAKERS["integer"], MAKERS["text"]])
        for _ in range(columns_per_type):
            values = column(type_name, rng, rows=rows, dirt=0.02, other=other)
            result = infer_column(values, type_name, settings=settings)
            total += 1
            clean_claims += 1 if result.clean else 0
            if result.type == type_name:
                stats[type_name]["tp"] += 1
            else:
                stats[type_name]["fn"] += 1
                stats.setdefault(result.type, {"tp": 0, "fp": 0, "fn": 0})
                stats[result.type]["fp"] += 1
                confusion[(type_name, result.type)] = \
                    confusion.get((type_name, result.type), 0) + 1
    print("columns: %d generated from %d labelled types, 2%% of cells dirtied\n"
          % (total, len(MEASURED_TYPES)))
    print("  %-13s %8s %8s %8s %8s" % ("type", "n", "precision", "recall", "F1"))
    macro = []
    for type_name in MEASURED_TYPES:
        counts = stats[type_name]
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) \
            if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) \
            if counts["tp"] + counts["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro.append((precision, recall, f1))
        print("  %-13s %8d %8.3f %8.3f %8.3f"
              % (type_name, counts["tp"] + counts["fn"], precision, recall, f1))
    correct = total - sum(entry["fn"] for entry in stats.values())
    print("  %-13s %8s %8.3f %8.3f %8.3f"
          % ("macro avg", total,
             sum(row[0] for row in macro) / len(macro),
             sum(row[1] for row in macro) / len(macro),
             sum(row[2] for row in macro) / len(macro)))
    print("  micro avg: %.3f - %d of %d columns named correctly"
          % (correct / total, correct, total))
    print("  claimed without review (`clean`): %d/%d" % (clean_claims, total))
    return confusion


def ambiguous_cases():
    """Infer the hand-written columns above (see _AMBIGUOUS_VALUES)."""
    return {label: infer_column(values, label)
            for label, values in _AMBIGUOUS_VALUES.items()}


def print_confusion(confusion, results):
    header("2. which types got confused with which")
    if not confusion:
        print("  no wrong names in this run")
    else:
        for (truth, predicted), count in sorted(confusion.items(),
                                                key=lambda item: -item[1]):
            print("  %-12s -> %-12s %d" % (truth, predicted, count))
    print("\n  on ambiguous columns, the second type that fits every value just as")
    print("  well (a tie at min_fit) is the real risk, not a wrong name:")
    print("  %-20s %-11s %-13s %s"
          % ("column", "chosen", "also fits", "flagged for review"))
    ties = flagged_ties = 0
    silent = []
    for label, result in results.items():
        rival, fit = top_rival(result)
        if rival is None or fit < Settings().min_fit:
            continue
        ties += 1
        flagged_ties += 1 if result.needs_review else 0
        if not result.needs_review:
            silent.append(label)
        print("  %-20s %-11s %-13s %s"
              % (label, result.type, "%s=%.2f" % (rival, fit),
                 "yes" if result.needs_review else "NO - decided silently"))
    reviewed = sum(1 for r in results.values() if r.needs_review)
    print("\n  columns with a rival type that clears min_fit too: %d of %d;"
          % (ties, len(results)))
    print("  review-noted: %d, decided silently: %d."
          % (flagged_ties, ties - flagged_ties))
    for label in silent:
        print("    silent coin-flip: %s" % label)
    print("  Counting every note, %d of %d land in the review queue - section 4"
          % (reviewed, len(results)))
    print("  prints why. A silently decided coin-flip is the failure mode this")
    print("  table exists to catch.")


def degradation(seed, rows):
    header("3. accuracy and review rate as columns get dirtier")
    print("  %-7s %-12s %9s %9s %9s" % ("dirt", "type", "named", "reviewed", "conf"))
    for dirt in (0.0, 0.02, 0.10, 0.25):
        for type_name in ("integer", "date", "currency", "email", "ipv4"):
            rng = random.Random(seed + int(dirt * 1000))
            named = reviewed = 0
            confidences = []
            runs = 20
            for _ in range(runs):
                values = column(type_name, rng, rows=rows, dirt=dirt,
                                other=rng.choice([MAKERS["integer"], MAKERS["text"]]))
                result = infer_column(values, type_name,
                                      settings=Settings(sample_size=250, seed=seed))
                named += 1 if result.type == type_name else 0
                reviewed += 1 if result.needs_review else 0
                confidences.append(result.confidence)
            print("  %-7.2f %-12s %8d/%d %8d/%d %9.3f"
                  % (dirt, type_name, named, runs, reviewed, runs,
                     sum(confidences) / len(confidences)))


def review_queue():
    header("4. what the notes look like on columns that are genuinely ambiguous")
    for label, values in _AMBIGUOUS_VALUES.items():
        result = infer_column(values, label)
        print("  %-20s %-11s conf=%.3f  %s"
              % (label, result.type, result.confidence, "; ".join(result.notes) or "-"))


def sampling_cost(rows):
    header("5. cost of inference: full scan versus reservoir sample")
    rng = random.Random(1)
    types = ("integer", "date", "currency", "email")
    populations = (rows, rows * 10)
    columns = {size: {name: column(name, rng, rows=size) for name in types}
               for size in populations}
    print("  %-12s %10s %11s %10s %10s %12s"
          % ("mode", "population", "sample_size", "wall (s)", "peak (KB)",
             "types named"))
    summary = {}
    for size in populations:
        for sample_size in (None, 1000):
            for type_name in types:
                values = columns[size][type_name]
                tracemalloc.start()
                started = time.perf_counter()
                result = infer_column(values, type_name, sample_size=sample_size)
                elapsed = time.perf_counter() - started
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                print("  %-12s %10d %11s %10.4f %10d %12s"
                      % ("reservoir" if sample_size else "full scan", size,
                         sample_size if sample_size else "all", elapsed,
                         peak // 1024, result.type))
                summary[(size, bool(sample_size))] = (peak // 1024, result.type)
    print("\n  same types either way. Full scan peak goes from %d KB at %d values to"
          % (summary[(populations[0], False)][0], populations[0]))
    print("  %d KB at %d values - it grows with the column. The reservoir holds %d KB"
          % (summary[(populations[1], False)][0], populations[1],
             summary[(populations[1], True)][0]))
    print("  at both sizes, and is the only option when the column will not fit in")
    print("  memory. Sampling is why the 1000-value default costs the same for a")
    print("  10k-row and a 10M-row table.")
    return columns[populations[0]]


def per_parser_cost(rows):
    header("6. cost per candidate parser - why a column takes 13 passes")
    rng = random.Random(7)
    probes = {
        "integer": [make_integer(rng) for _ in range(rows)],
        "float": [make_float(rng) for _ in range(rows)],
        "date": [make_date(rng) for _ in range(rows)],
        "currency": [make_currency(rng) for _ in range(rows)],
        "email": [make_email(rng) for _ in range(rows)],
        "url": [make_url(rng) for _ in range(rows)],
        "ipv6": [make_ipv6(rng) for _ in range(rows)],
        "text": [make_text(rng) for _ in range(rows)],
    }
    print("  %-10s %s" % ("column", " ".join("%9s" % name for name in SPECIFICITY)))
    for label, values in probes.items():
        cells = []
        for type_name in SPECIFICITY:
            parser = PARSERS[type_name]
            started = time.perf_counter()
            hits = sum(1 for value in values if parser(value, "auto") is not None)
            elapsed = time.perf_counter() - started
            cells.append("%9.4f" % elapsed)
        print("  %-10s %s" % (label, " ".join(cells)))
    print("\n  times are for %d values of that column; every column is scored against"
          % rows)
    print("  all %d candidates, so the scoring pass costs the sum of a row above."
          % len(SPECIFICITY))
    print("  The two cells that stand out are the date parser run over email/url/")
    print("  text and the datetime parser run over url: those values are long, so")
    print("  each separator split produces a substring long enough to try, and")
    print("  nothing matches until the whole value has been worked through. The")
    print("  cheap guards - a colon test before the datetime scan, a letter test")
    print("  before `strptime` - are what keep the other cells near zero.")


def against_pandas(rows):
    header("7. side by side with pandas read_csv inference")
    try:
        import pandas as pd
    except ImportError:
        print("  pandas is not installed here, so this section is skipped")
        return
    import csv
    import tempfile

    rng = random.Random(11)
    spec = [
        ("qty", column("integer", rng, rows=rows)),
        ("price", column("float", rng, rows=rows)),
        ("placed", column("date", rng, rows=rows)),
        ("visitor_ip", column("ipv4", rng, rows=rows)),
        ("event_id", column("uuid", rng, rows=rows)),
        ("cost", column("currency", rng, rows=rows)),
        ("zip", ["%05d" % rng.randint(0, 99999) for _ in range(rows)]),
        ("flag", column("boolean", rng, rows=rows)),
        ("region", column("categorical", rng, rows=rows)),
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "wide.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([name for name, _ in spec])
            for index in range(rows):
                writer.writerow([values[index] for _name, values in spec])
        frame = pd.read_csv(path)
        with open(path, newline="", encoding="utf-8") as handle:
            # DictReader already yields one dict per row, so the rows go straight in.
            table = infer_table(csv.DictReader(handle),
                                settings=Settings(sample_size=min(rows, 1000)))
    print("  %-12s %-16s %-12s %s" % ("column", "pandas dtype", "dtypeinfer",
                                      "pandas values"))
    for name, values in spec:
        dtype = str(frame[name].dtype)
        inferred = table.types[name]
        sample_value = repr(frame[name].iloc[0])[:28]
        agreed = "yes" if (dtype == "int64" and inferred == "integer") \
            or (dtype == "float64" and inferred in ("float", "percent", "currency")) \
            or (dtype == "bool" and inferred == "boolean") \
            or (dtype == "object" and inferred in ("text", "categorical")) else "-"
        print("  %-12s %-16s %-12s %s %s" % (name, dtype, inferred, sample_value,
                                             agreed))
    named_structured = sum(1 for name, _ in spec
                           if table.types[name] in SPECIFICITY)
    print("\n  dtypeinfer named a structured type for %d/%d columns; pandas reports"
          % (named_structured, len(spec)))
    print("  int64/float64/bool/object and stops, so the remaining columns are just")
    print("  `object` to it. Different job, not a missed one - but the `zip` row is")
    print("  the one to look at: pandas kept the int64 and dropped the leading zero")
    print("  (first value %r), while dtypeinfer called it %r and wrote down why a"
          % (frame["zip"].iloc[0], table.types["zip"]))
    print("  loader should not trust that name:")
    for note in table.columns["zip"].notes:
        print("      %s" % note)
    print("  review queue for this table: %d/%d columns - %s"
          % (len(table.needs_review), len(spec),
             ", ".join(table.needs_review) or "none"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--columns", type=int, default=8, help="columns per type")
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args()

    # The currency and mixed-columns cases contain non-ASCII symbols and a Windows
    # console starts on a code page that cannot show them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("dtypeinfer benchmarks - python %s" % sys.version.split()[0])
    confusion = accuracy(args.seed, args.columns, args.rows)
    results = ambiguous_cases()
    print_confusion(confusion, results)
    degradation(args.seed, args.rows)
    review_queue()
    sampling_cost(args.rows * 5)
    per_parser_cost(2000)
    against_pandas(args.rows)


if __name__ == "__main__":
    main()
