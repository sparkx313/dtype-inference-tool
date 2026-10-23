"""Infer the type of a column, and say what is uncertain about it.

The decision itself is deliberately boring: score every structured type against
the sample, take the most specific one that clears `min_fit`, and if nothing
clears it choose between `mixed`, `categorical` and `text` on how the values
actually distributed. The interesting part is what comes back alongside the type -
the fit of every candidate, the spellings found, and the ambiguities no single
value can settle.
"""

from collections import Counter, OrderedDict

from . import ambiguity, parse
from .sample import ColumnSampler, reservoir_sample
from .types import PARSERS, PYTHON_TYPES, SPECIFICITY, Settings

# A type that is only ever reported because a more specific one also fitted, so
# `integer` fitted does not make a float column look ambiguous.
IMPLIED_BY = {"integer": frozenset({"float"})}


class ColumnInference:
    """One column's inferred type plus everything the sample showed.

    `confidence` is the share of non-empty values the type accepted; `clean` says
    that share reached `min_fit`, so a column of integers with three text values in
    it still gets named `integer` - it just does not get believed silently.
    """

    __slots__ = ("name", "type", "confidence", "clean", "fits", "null_ratio",
                 "population", "sampled", "non_null", "distinct",
                 "cardinality_ratio", "notes", "tags", "rejects", "unfit_examples",
                 "top_values", "decimal_style", "decimal_evidence", "python_type",
                 "null_values")

    def __init__(self, **fields):
        for slot in self.__slots__:
            setattr(self, slot, fields.get(slot))

    @property
    def ambiguity_notes(self):
        return [note for note in self.notes if note.split(":")[0] in AMBIGUOUS_NOTES]

    @property
    def needs_review(self):
        return not self.clean or bool(self.ambiguity_notes)

    def as_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self):
        return (f"ColumnInference({self.name!r}: {self.type} "
                f"conf={self.confidence:.3f} notes={self.notes})")


AMBIGUOUS_NOTES = frozenset({
    "ambiguous_three_digit_tail", "decimal_separator_inconsistent",
    "order_ambiguous", "two_digit_year", "timezone_offset_dropped",
    "leading_zeros", "identifier_like", "boolean_like", "also_fits",
})


def infer_column(values, name="", settings=None, population=None, **overrides):
    """Infer one column's type from an iterable of cell values.

    `values` is consumed once, so a generator is fine. With `sample_size` set the
    scan still touches every value to count it, but only `sample_size` of them are
    kept. Pass `population` when the caller already knows the true row count and
    is handing over a pre-made sample.
    """
    config = _settings(settings, overrides)
    counted = _Counted()
    if config.sample_size:
        raw_sample = reservoir_sample(_count(values, counted), config.sample_size,
                                      config.seed)
    else:
        raw_sample = list(_count(values, counted))
    sample = [parse.clean(value) for value in raw_sample]
    total = counted.total if population is None else population
    extra_nulls = frozenset(token.lower() for token in config.null_values)

    non_null = [value for value in sample if not parse.as_null(value, extra_nulls)]
    stats = {
        "population": total,
        "sampled": len(sample),
        "non_null_count": len(non_null),
        "null_ratio": 1.0 - (len(non_null) / len(sample)) if sample else 0.0,
    }
    if not non_null:
        return _build(name, "empty", 1.0, {}, stats, [], config.decimal_style, {},
                      config, sample)

    if config.decimal_style == "auto":
        style, evidence = ambiguity.resolve_decimal_style(non_null)
    else:
        style, evidence = config.decimal_style, {}
    fits, tags_by_type, rejects, unfitted = _score(non_null, style, config)
    chosen, confidence = _choose(fits, non_null, config)

    distinct = len(set(non_null))
    chosen_tags = tags_by_type.get(chosen, [])
    stats.update({
        "distinct": distinct,
        "cardinality_ratio": distinct / len(non_null),
        "ipv4_like": _ipv4_like(non_null, fits),
        "decimal_style": style,
        "decimal_evidence": evidence,
        "unfit": len(unfitted),
    })
    notes = ambiguity.build_notes(non_null, chosen, chosen_tags, fits, stats, config)
    result = _build(name, chosen, confidence, fits, stats, notes, style, evidence,
                    config, non_null)
    result.tags = dict(Counter(chosen_tags))
    result.rejects = rejects
    result.unfit_examples = unfitted[:config.example_rejects]
    return result


def infer_table(rows, columns=None, settings=None, **overrides):
    """Infer every column of an iterable of mappings in one pass over the rows.

    Each column gets its own reservoir, so peak memory is `columns * sample_size`
    values regardless of how long the table is.
    """
    config = _settings(settings, overrides)
    sampler = ColumnSampler(k=config.sample_size or _UNBOUNDED, seed=config.seed)
    for row in rows:
        if columns is not None:
            row = {name: row[name] for name in columns if name in row}
        sampler.add(row)
    inference = OrderedDict()
    for column, values in sampler.samples.items():
        inference[column] = infer_column(
            values, column,
            settings=config.replace(sample_size=None),
            population=sampler.counts[column])
    return TableInference(inference, sampler.rows, config)


_UNBOUNDED = 1 << 30


class TableInference:
    __slots__ = ("columns", "population", "settings")

    def __init__(self, columns, population, settings):
        self.columns = columns
        self.population = population
        self.settings = settings

    @property
    def types(self):
        return {name: column.type for name, column in self.columns.items()}

    @property
    def needs_review(self):
        return [name for name, column in self.columns.items() if column.needs_review]

    def summary(self):
        counts = Counter(column.type for column in self.columns.values())
        return {
            "columns": len(self.columns),
            "population": self.population,
            "types": dict(counts),
            "needs_review": len(self.needs_review),
        }

    def __repr__(self):
        return f"TableInference({self.types})"


def _settings(settings, overrides):
    if settings is None:
        return Settings(**overrides)
    return settings.replace(**overrides) if overrides else settings


def _score(non_null, style, config):
    """Fit ratio, spelling tags and example rejects for every candidate type."""
    total = len(non_null)
    fits = {}
    tags_by_type = {}
    rejects = {}
    any_hit = bytearray(total)
    for type_name in SPECIFICITY:
        parser = PARSERS[type_name]
        hits = 0
        tags = []
        misses = []
        for index, value in enumerate(non_null):
            parsed = parser(value, style)
            if parsed is None:
                if len(misses) < config.example_rejects:
                    misses.append(value)
            else:
                hits += 1
                any_hit[index] = 1
                tags.append(parsed[1])
        fits[type_name] = hits / total
        tags_by_type[type_name] = tags
        if misses:
            rejects[type_name] = misses
    unfitted = [value for index, value in enumerate(non_null) if not any_hit[index]]
    return fits, tags_by_type, rejects, unfitted


def _choose(fits, non_null, config):
    """Pick a type, or report that the column is more than one type.

    A type clearing `min_fit` is claimed outright. One that dominates but falls
    short is still reported, with its confidence below 1.0 and the values that did
    not fit, because "integer with three bad cells" is more useful than "mixed".
    """
    for type_name in config.ordered_types:
        if fits[type_name] >= config.min_fit:
            return type_name, fits[type_name]
    candidates = [name for name in SPECIFICITY if fits[name] >= config.mixed_floor]
    independent = [
        name for name in candidates
        if not any(name in IMPLIED_BY.get(other, frozenset()) for other in candidates
                   if other != name)
    ]
    if len(independent) >= 2:
        best = max(independent, key=lambda name: fits[name])
        return "mixed", fits[best]
    if len(independent) == 1:
        only = independent[0]
        return only, fits[only]
    distinct = len(set(non_null))
    if len(non_null) >= config.categorical_min_values \
            and distinct <= config.categorical_max_distinct \
            and distinct / len(non_null) <= config.categorical_ratio:
        return "categorical", 1.0
    return "text", 1.0


def _ipv4_like(non_null, fits):
    if fits.get("ipv4", 0.0) <= 0.0:
        return 0
    return sum(1 for value in non_null
               if parse.as_ipv4(value) is None and parse.looks_like_ipv4(value))


def _build(name, chosen, confidence, fits, stats, notes, style, evidence, config,
           values):
    result = ColumnInference()
    result.name = name
    result.type = chosen
    result.confidence = confidence
    result.clean = confidence >= config.min_fit
    result.fits = dict(fits)
    result.null_ratio = stats["null_ratio"]
    result.population = stats["population"]
    result.sampled = stats["sampled"]
    result.non_null = stats["non_null_count"]
    result.distinct = stats.get("distinct", len(set(values)))
    result.cardinality_ratio = stats.get("cardinality_ratio", 0.0)
    result.notes = notes
    result.tags = {}
    result.rejects = {}
    result.unfit_examples = []
    result.top_values = Counter(values).most_common(config.top_values)
    result.decimal_style = style
    result.decimal_evidence = evidence
    result.python_type = PYTHON_TYPES[chosen]
    result.null_values = frozenset(token.lower() for token in config.null_values)
    return result


class _Counted:
    __slots__ = ("total",)

    def __init__(self):
        self.total = 0


def _count(values, counted):
    for value in values:
        counted.total += 1
        yield value
