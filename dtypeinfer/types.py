"""The types the inferrer can name, and one parser for each.

A parser is used for both questions at once: it answers "does this value fit the
type" (non-None means yes) and "what is the value" (so coercion never re-decides
what inference already decided).
"""

from dataclasses import dataclass, field

from . import parse

# Ordered from most specific to least specific. When a column fits more than one
# type, the first one that clears the fit threshold wins, so a column of `1.234`
# is never called a float while `integer` is still on the table, and an 8-digit
# compact date is never called an integer.
SPECIFICITY = (
    "uuid",
    "email",
    "ipv6",
    "ipv4",
    "url",
    "datetime",
    "date",
    "time",
    "percent",
    "currency",
    "boolean",
    "integer",
    "float",
)

# Types the inferrer can report that have no per-value parser, because they
# describe the shape of a column rather than the shape of a value.
COLUMN_TYPES = ("empty", "mixed", "categorical", "text")


def parse_uuid(text, style="auto"):
    return parse.as_uuid(text)


def parse_email(text, style="auto"):
    return parse.as_email(text)


def parse_ipv6(text, style="auto"):
    return parse.as_ipv6(text)


def parse_ipv4(text, style="auto"):
    return parse.as_ipv4(text)


def parse_url(text, style="auto"):
    return parse.as_url(text)


def parse_datetime(text, style="auto"):
    return parse.as_datetime(text)


def parse_date(text, style="auto"):
    return parse.as_date(text)


def parse_time(text, style="auto"):
    return parse.as_time(text)


def parse_percent(text, style="auto"):
    return parse.as_percent(text, style)


def parse_currency(text, style="auto"):
    return parse.as_currency(text, style)


def parse_boolean(text, style="auto"):
    return parse.as_boolean(text)


def parse_integer(text, style="auto"):
    return parse.as_integer(text, style)


def parse_float(text, style="auto"):
    return parse.as_float(text, style)


def parse_text(text, style="auto"):
    cleaned = parse.clean(text)
    return None if cleaned is None else (cleaned, "text")


def parse_mixed(text, style="auto"):
    cleaned = parse.clean(text)
    return None if cleaned is None else (cleaned, "mixed")


PARSERS = {
    "uuid": parse_uuid,
    "email": parse_email,
    "ipv6": parse_ipv6,
    "ipv4": parse_ipv4,
    "url": parse_url,
    "datetime": parse_datetime,
    "date": parse_date,
    "time": parse_time,
    "percent": parse_percent,
    "currency": parse_currency,
    "boolean": parse_boolean,
    "integer": parse_integer,
    "float": parse_float,
    "text": parse_text,
    "mixed": parse_mixed,
}

# What a coerced value of each type actually is, for callers that check the
# result type rather than read the documentation.
PYTHON_TYPES = {
    "uuid": "uuid.UUID",
    "email": "str",
    "ipv4": "ipaddress.IPv4Address",
    "ipv6": "ipaddress.IPv6Address",
    "url": "str",
    "datetime": "datetime.datetime",
    "date": "datetime.date",
    "time": "datetime.time",
    "percent": "float",
    "currency": "float",
    "boolean": "bool",
    "integer": "int",
    "float": "float",
    "text": "str",
    "categorical": "str",
    "mixed": "str",
    "empty": "NoneType",
}


@dataclass(frozen=True)
class Settings:
    """Every knob the inferrer has, in one place, so a result can be reproduced.

    `min_fit` is the share of non-empty values that must conform for a type to be
    claimed at all; `mixed_floor` is the share at which a type that clears neither
    still gets named, and two types clearing it together make the column `mixed`.
    A column with no structured fit is `categorical` when it holds at most
    `categorical_max_distinct` different values in `categorical_ratio` of its
    sample, and `text` otherwise.
    """

    min_fit: float = 0.95
    mixed_floor: float = 0.5
    cardinality_note_ratio: float = 0.5
    categorical_ratio: float = 0.5
    categorical_max_distinct: int = 20
    categorical_min_values: int = 8
    sample_size: int = 1000
    seed: int = 0
    decimal_style: str = "auto"
    null_values: tuple = ()
    example_rejects: int = 3
    top_values: int = 5

    @property
    def ordered_types(self):
        return SPECIFICITY

    def replace(self, **changes):
        return Settings(**{**self._as_dict(), **changes})

    def _as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
