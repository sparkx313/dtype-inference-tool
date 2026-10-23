"""Column-level judgement: what a single value cannot tell you.

`1.234` is either a grouped thousand or a decimal point, and no amount of staring
at that one value settles it. Across a column the answer is usually forced - one
`12,5` anywhere proves the author writes decimal commas, and then every
three-digit tail has to be grouping. This module makes that decision, states it in
a note, and lists the other ambiguities a reviewer should look at.
"""

import re

from . import parse
from .types import SPECIFICITY

THREE_DIGIT_TAIL = re.compile(r"[+-]?\d{1,3}[.,]\d{3}\Z")
NUMERIC_TYPES = frozenset({"integer", "float", "percent", "currency"})
DATE_TYPES = frozenset({"date", "datetime", "time"})
AMBIGUOUS_ORDER_TAGS = ("order_ambiguous", "two_digit_year",
                        "timezone_offset_dropped", "named_month")

# Fitting `integer` always means `float` fits too, so that pairing is not an
# ambiguity worth a note. Anything else that clears the threshold is.
SUPERSETS = {"integer": frozenset({"float"})}


def proves_comma_decimal(body):
    if "," not in body:
        return False
    if "." in body and body.rfind(",") < body.rfind("."):
        return False
    head, _, tail = body.rpartition(",")
    return len(tail) != 3 or len(head.lstrip("+-")) > 3


def proves_dot_decimal(body):
    if body.count(".") > 1:
        return False
    head, _, tail = body.rpartition(".")
    return bool(tail) and (len(tail) != 3 or len(head.lstrip("+-")) > 3)


def resolve_decimal_style(values):
    """`("eu" | "us" | "mixed", evidence)` read off the whole sample.

    The style only changes how an ambiguous `1.234` is coerced, but it changes it
    for every value in the column, so the evidence is worth returning.
    """
    comma = dot = dot_tail = comma_tail = 0
    for value in values:
        body = parse.clean(value)
        if body is None or not any(char in body for char in ".,"):
            continue
        if THREE_DIGIT_TAIL.match(body):
            if body[-4] == ".":
                dot_tail += 1
            else:
                comma_tail += 1
        elif proves_comma_decimal(body):
            comma += 1
        elif proves_dot_decimal(body):
            dot += 1
    evidence = {
        "decimal_comma": comma,
        "decimal_point": dot,
        "three_digit_tail_after_dot": dot_tail,
        "three_digit_tail_after_comma": comma_tail,
    }
    if comma and dot:
        return "mixed", evidence
    if comma:
        return "eu", evidence
    return "us", evidence


def count_tag(tags, name):
    return sum(1 for tag in tags if name in tag.split("+"))


def build_notes(values, chosen, tags, fits, stats, settings):
    """Strings shaped `name` or `name:payload`, so tests can assert on them."""
    notes = []
    if stats["null_ratio"] > 0:
        notes.append(f"has_nulls:{stats['null_ratio']:.2f}")
    if stats["sampled"] < stats["population"]:
        notes.append(f"sampled:{stats['sampled']}/{stats['population']}")
    if stats.get("unfit") and chosen in SPECIFICITY:
        # Values that no candidate type accepted at all: the honest reason the
        # confidence is below 1.0, next to examples in `result.rejects`. For a
        # `text` or `categorical` column that is the definition, not a defect.
        notes.append(f"unfit_values:{stats['unfit']}")

    evidence = stats["decimal_evidence"]
    if chosen in NUMERIC_TYPES:
        leading = count_tag(tags, "leading_zero")
        if leading:
            notes.append(f"leading_zeros:{leading}")
            notes.append("identifier_like")
        commas = count_tag(tags, "decimal_comma")
        if commas:
            notes.append(f"decimal_comma:{commas}")
        dot_tail = evidence.get("three_digit_tail_after_dot", 0)
        comma_tail = evidence.get("three_digit_tail_after_comma", 0)
        if dot_tail or comma_tail:
            # A three-digit tail is the only spelling the column style actually
            # changes, so the note reports both counts and the style chosen.
            notes.append(
                f"ambiguous_three_digit_tail:dot={dot_tail},comma={comma_tail},"
                f"style={stats['decimal_style']}")
        if stats["decimal_style"] == "mixed":
            notes.append("decimal_separator_inconsistent")
        if chosen == "integer" and set(values) <= {"0", "1"}:
            notes.append("boolean_like")

    if chosen in DATE_TYPES:
        for tag in AMBIGUOUS_ORDER_TAGS:
            found = count_tag(tags, tag)
            if found:
                notes.append(f"{tag}:{found}")

    if chosen == "ipv4" and stats["ipv4_like"]:
        notes.append(f"leading_zero_octets_rejected:{stats['ipv4_like']}")

    implied = SUPERSETS.get(chosen, frozenset())
    secondary = [
        f"{name}={fits[name]:.2f}"
        for name in settings.ordered_types
        if name != chosen and name not in implied and fits.get(name, 0.0) >= settings.min_fit
    ]
    if secondary:
        notes.append("also_fits:" + ",".join(secondary))

    small_set = (len(values) >= settings.categorical_min_values
                 and stats["cardinality_ratio"] <= settings.cardinality_note_ratio)
    if small_set and chosen not in ("categorical", "empty"):
        notes.append(f"low_cardinality:{stats['distinct']}/{len(values)}")
    return notes
