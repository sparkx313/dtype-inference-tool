# dtypeinfer — column type inference that shows its working

`dtypeinfer` reads a CSV (or any iterable of row mappings) one pass at a time and
names a type for every column: `date`, `ipv4`, `currency`, `uuid`, `integer`, …
It differs from `pandas.read_csv`'s dtype guess in one structural way: **when the
spelling of a value cannot settle a question, it says so instead of guessing
silently.**

`1.234` is either one-thousand-two-hundred-thirty-four or one point two three
four. `03/04/2020` is either the 3rd of April or the 4th of March. `02134` is an
integer that stops being one the moment you load it. No amount of looking at a
*single* cell resolves any of those — only the rest of the column can, and even
then some of them stay undecidable. A tool that has to answer anyway should hand
back the answer *and* the reason you should look at it. That is what this is.

Pure standard library. No dependencies, no C extensions, no download step.

## Run it

```
$ git clone <this repo> && cd dtype-inference-tool
$ python -m unittest discover -s tests -p 'test_*.py' -t tests
Ran 59 tests in 0.129s

OK
$ python benchmarks/run_benchmarks.py          # ~37 s, prints the tables below
```

The package is a directory, so it works from a clone with `sys.path` pointed at
the repo root; `pyproject.toml` is there if you would rather install it.

## Using it

```python
from dtypeinfer import infer_column, infer_csv, coerce_column

values = ["02134", "10001", "90210", "00501", "19103"] * 4
result = infer_column(values, "zip")

print(result)
# ColumnInference('zip': integer conf=1.000
#                 notes=['leading_zeros:8', 'identifier_like', 'low_cardinality:5/20'])
print(result.needs_review)   # True - a clean name is not the same as a safe one

coerce_column(values, result, on_error="null").values[:4]
# [2134, 10001, 90210, 501]  <- the leading zeros are gone; that is the whole warning
```

A whole file, streaming, with a European decimal style resolved from the data:

```python
from dtypeinfer import infer_csv
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "orders.csv")
with open(path, "w", newline="", encoding="utf-8") as handle:
    handle.write("id;when;total\n1;2026-01-05;12,50\n2;2026-02-11;7,30\n3;;99,99\n")

# ';' is the field separator here, so ',' is free to mean the decimal mark.
table = infer_csv(path, decimal_style="eu")

print(table.types)
# {'id': 'integer', 'when': 'date', 'total': 'float'}
print(table.summary())
# {'columns': 3, 'population': 3, 'types': {'integer': 1, 'date': 1, 'float': 1},
#  'needs_review': 0}
print(table.columns["when"].notes)      # ['has_nulls:0.33']
print(table.columns["total"].notes)     # ['decimal_comma:3']
print(list(table.needs_review))         # [] - nothing here needs a human eye
```

## What you get back

`infer_column` returns a `ColumnInference`; the fields that matter:

| field | meaning |
| --- | --- |
| `type` | the named type — a structured one, or `categorical` / `text` / `mixed` / `empty` |
| `confidence` | share of non-empty sampled values the type accepted |
| `clean` | `confidence >= min_fit` (default 0.95): the claim needs no arguing |
| `needs_review` | `not clean` **or** an ambiguity note fired |
| `fits` | the full profile: every one of the 13 structured types with its fit ratio |
| `notes` | the machine-readable list of every doubt, with counts (`order_ambiguous:60`) |
| `tags` | spelling tags seen while parsing (`day_first`, `thousands_separator`, …) |
| `unfit_examples` | up to 3 values that did not fit, with positions |
| `decimal_style` | `us` / `eu` / `mixed` as resolved from the column |
| `population`, `sampled`, `distinct`, `cardinality_ratio` | what was actually looked at |

`fits` is the part worth having: a `date` column that also fits `integer` at 1.00
(8-digit compact dates) is *visible* as a coin-flip, and `also_fits:integer=1.00`
lands in `notes`, so nothing is decided behind your back.

## The note vocabulary

Every note is `<name>[:count]`, and the ambiguous ones are in
`infer.AMBIGUOUS_NOTES` — that set is what turns `needs_review` on:

| note | when it fires |
| --- | --- |
| `leading_zeros:n` + `identifier_like` | numeric column whose values start with `0` — a code, not a quantity |
| `boolean_like` | an integer column containing only `0` and `1` |
| `ambiguous_three_digit_tail:dot=N,comma=M,style=S` | `1.234` vs `1,234`: the only spelling the column's style actually changes |
| `decimal_comma:n`, `decimal_separator_inconsistent` | European decimals, and a column that mixes both marks |
| `order_ambiguous:n` | `03/04/2020` — every value fits both day-first and month-first |
| `two_digit_year:n` | `1/2/20` — which century is a guess |
| `timezone_offset_dropped:n` | `+02:00` parsed and then thrown away |
| `also_fits:type=ratio` | another type that explains the column equally well |
| `unfit_values:n` + `leading_zero_octets_rejected:n` | values the chosen type did not accept |
| `has_nulls:ratio`, `sampled:k/n`, `low_cardinality:d/n` | descriptive, never triggers review |

## Measurements

`python benchmarks/run_benchmarks.py` — 120 synthetic columns over 15 labelled
types, 2% of cells dirtied, seed `20261023`. Full output is in
[`benchmarks/last_run.txt`](benchmarks/last_run.txt).

**1. Naming accuracy.** 120/120 columns named correctly; precision, recall and F1
are 1.000 for every type; 112/120 claimed without review.

Read that as the *floor*, not the headline: the generator writes one canonical
spelling per type, so there is nothing to get wrong. What this corpus does measure
is that the 8 flagged columns were flagged for a real reason and never renamed.
The interesting number is section 2:

**2. Ties.** Over 14 hand-written genuinely ambiguous columns, exactly **1** has a
second type that clears `min_fit` too — `20260131` fits `date` and `integer` at
1.00 — and it is review-noted (`also_fits:integer=1.00,float=1.00`). **0 silent
coin-flips.** 8 of the 14 land in the review queue.

**3. Degradation.** 20 runs per cell, 400 values, sample 250:

| dirt | type | named | reviewed | mean conf |
| --- | --- | --- | --- | --- |
| 0.00 | integer / date / currency / email / ipv4 | 20/20 each | 0/20 | 1.000 |
| 0.02 | integer | 20/20 | 0/20 | 0.998 |
| 0.02 | date / currency / email / ipv4 | 20/20 | 0/20 | 0.992–0.994 |
| 0.10 | integer | 20/20 | 2/20 | 0.981 |
| 0.10 | date / currency / email / ipv4 | 20/20 | 0–2/20 | 0.967–0.968 |
| 0.25 | integer | 20/20 | 11/20 | 0.949 |
| 0.25 | date | 20/20 | 20/20 | 0.912 |
| 0.25 | currency / email / ipv4 | 20/20 | 19–20/20 | 0.913–0.915 |

The name survives a quarter of the column being junk; what changes is that the
tool stops pretending it is clean. That is the intended behaviour: "integer with
3 bad cells" beats "mixed".

**4. Notes on hard columns** (verbatim from the run):

```
zip codes            integer     conf=1.000  leading_zeros:40; identifier_like
flag column          integer     conf=1.000  boolean_like
eu decimals          float       conf=1.000  decimal_comma:60
ambiguous tail       float       conf=1.000  ambiguous_three_digit_tail:dot=60,comma=0,style=us
slash dates          date        conf=1.000  order_ambiguous:60
unpadded dates       date        conf=1.000  order_ambiguous:40; two_digit_year:40
compact dates        date        conf=1.000  also_fits:integer=1.00,float=1.00
sparse dates         date        conf=1.000  has_nulls:0.60
leading-zero ip      ipv4        conf=0.667  unfit_values:20; leading_zero_octets_rejected:20
inconsistent decimals float      conf=1.000  decimal_comma:20; decimal_separator_inconsistent
```

**5. Cost.** `tracemalloc` peak and wall time, one column:

| mode | population | sample | wall (s) | peak (KB) |
| --- | --- | --- | --- | --- |
| full scan | 2 000 | all | 0.15–1.41 | 226–403 |
| reservoir | 2 000 | 1000 | 0.07–0.57 | 77–166 |
| full scan | 20 000 | all | 1.49–11.58 | **1 319–5 001** |
| reservoir | 20 000 | 1000 | 0.14–0.65 | **77–166** |

Same type named either way. A full scan grows with the column; a reservoir costs
the same at 20 000 values as at 2 000, which is the only version that works on a
file that does not fit in memory. Peak is `columns × sample_size` values.

**6. Parser cost.** 2 000 values per column × 13 candidate parsers = 104 cells;
83 of them are under 5 ms. The two hot spots are the `date` parser run over
email/URL/text (0.13–0.18 s — long values, every separator split gets a try) and
`datetime` over URLs (0.44 s, because `https://` contains the colon the clock test
looks for). Two cheap guards keep everything else near zero: a `":" in value` test
before the datetime split scan, and a letter test before falling through to
`strptime` on named-month formats.

**7. Against pandas.** Same 9-column CSV, 400 rows:

| column | pandas dtype | dtypeinfer |
| --- | --- | --- |
| qty | int64 | integer |
| price | float64 | float |
| placed | object | date |
| visitor_ip | object | ipv4 |
| event_id | object | uuid |
| cost | object | currency |
| zip | int64 | integer + `leading_zeros:30`, `identifier_like` |
| flag | object | boolean |
| region | object | categorical |

dtypeinfer named a structured type for 8/9 columns; pandas answers
`int64/float64/bool/object` and stops, which is a different job, not a missed one.
The row to look at is `zip`: pandas returns `np.int64(12158)` — the leading zero is
already gone, silently — while dtypeinfer gives the same name *plus* the reason not
to load it as a number. Review queue for that table: 1/9 columns.

## How it decides

1. **Value parsers** (`parse.py`) never vote. Each returns `(value, tag)` or `None`,
   and the tag records *which reading was used* — `day_first`, `decimal_comma`,
   `grouped_thousands`. A parser that cannot decide between two readings returns
   both, and the column decides.
2. **Number grammar** (`NUMBER_SOURCE`) accepts `1.234,56`, `1,234.56`, `1234,56`,
   `.5`, `1e5` and SI space grouping (`1 234,56`, incl. U+00A0/U+202F). Which mark
   is the decimal separator is a per-column property (`decimal_style`), resolved
   from evidence, not from the locale. `as_integer` only fires when the value is
   *written* as a whole number — `1.000` stays a float, `1.234` read as dot-grouping
   becomes an integer.
3. **Dates** are parsed without `strptime` for numeric forms: year-first, compact,
   and order-free with explicit `day_first` / `month_first` / `order_ambiguous`
   tags, validated against `calendar.monthrange` (Feb 31 is rejected).
4. **Identification** is by specificity ladder — `uuid > email > ipv6 > ipv4 > url >
   datetime > date > time > percent > currency > boolean > integer > float` — so an
   IPv4 in a column that also looks numeric is claimed by the more specific type.
5. **Scoring** runs every candidate parser over every sampled value, so `fits` is a
   profile rather than a verdict, and `unfit_examples` can quote the exact cells.
6. **Sampling** is Algorithm R with a fixed seed: bounded memory, exact population
   count, reproducible. `sample_size=None` scans everything.
7. **Coercion** (`coerce_column`) reuses the parser and the resolved style, and its
   `on_error` policy (`keep`/`null`/`raise`) is a caller decision, not a silent one.

## Limits, honestly

- Value parsers are hand-written regexes, not full RFC grammar. Email takes
  dot-atom and quoted local parts but rejects internationalised domains
  (`michel@exämple.com` → no match); URL wants a scheme
  (`example.com/a` → no match); IPv4 refuses leading-zero octets (`01.2.3.4`) *on
  purpose* and reports them as `leading_zero_octets_rejected`. `2026-02-31` is
  rejected outright.
- 100% synthetic accuracy is not a claim about your data. Real columns are section
  4: correct name, visible doubt, `needs_review=True`.
- No locale database. `decimal_style="auto"` infers from the column; if the column
  is ambiguous, so is the answer, and the note says which style was used.
- Timezone offsets are parsed and dropped (`timezone_offset_dropped`). Nothing here
  is tz-aware.
- Single-threaded. The cost table is what you get.

## Layout

```
dtypeinfer/
  parse.py      value parsers -> (value, tag) or None; no column knowledge
  types.py      the type ladder, parser registry, Settings dataclass
  sample.py     reservoir sampling (Algorithm R) + per-column sampler
  infer.py      scoring, choosing, ColumnInference / TableInference
  ambiguity.py  decimal-style resolution and the note vocabulary
  coerce.py     type -> Python value, with on_error policies
  csvread.py    streaming CSV entry point, delimiter sniffing, BOM handling
tests/test_dtypeinfer.py    59 tests
benchmarks/run_benchmarks.py  the seven tables above
```

MIT — see [LICENSE](LICENSE).
