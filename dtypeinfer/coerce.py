"""Turn the inferred type back into Python values.

Coercion uses the same parser that scored the column, so a value that fitted the
type always converts, and one that did not is reported instead of silently
becoming a different type.
"""

from . import parse, types
from .types import PARSERS


class CoercionReport:
    __slots__ = ("values", "converted", "failed", "nulls")

    def __init__(self, values, converted, failed, nulls):
        self.values = values
        self.converted = converted
        self.failed = failed
        self.nulls = nulls

    @property
    def ok(self):
        return not self.failed

    def __repr__(self):
        return (f"CoercionReport(converted={self.converted}, nulls={self.nulls}, "
                f"failed={len(self.failed)})")


def coerce_value(text, type_name, decimal_style="auto"):
    """The Python value for one cell, or None when it is null or unparseable."""
    body = parse.clean(text)
    if body is None or parse.as_null(body):
        return None
    if type_name in ("text", "categorical", "mixed"):
        return body
    parsed = PARSERS[type_name](body, decimal_style)
    return None if parsed is None else parsed[0]


def coerce_column(values, inference, on_error="keep"):
    """Coerce a whole column against an existing `ColumnInference`.

    `on_error="keep"` puts the original text where the value did not fit;
    `"null"` puts None; `"raise"` refuses, for a load that must be clean.
    """
    if on_error not in ("keep", "null", "raise"):
        raise ValueError(f"unknown on_error policy: {on_error!r}")
    style = inference.decimal_style or "auto"
    nulls_seen = inference.null_values or ()
    out = []
    failed = []
    nulls = 0
    for position, value in enumerate(values):
        body = parse.clean(value)
        if body is None or parse.as_null(body, nulls_seen):
            nulls += 1
            out.append(None)
            continue
        coerced = coerce_value(body, inference.type, style)
        if coerced is None:
            failed.append((position, body))
            if on_error == "raise":
                raise ValueError(f"{inference.name}: value {body!r} at {position} "
                                 f"is not {inference.type}")
            out.append(None if on_error == "null" else body)
            continue
        out.append(coerced)
    return CoercionReport(out, len(out) - nulls - len(failed), failed, nulls)
