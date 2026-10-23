"""Turn one text value into the Python object its type implies.

A parser here returns ``None`` (this value is not of this type) or a
``(value, tag)`` pair. The tag matters as much as the value: ``1,234.5`` and
``1234,5`` are the same *type* written two different ways, and that difference
belongs in a note the caller can act on, not in a second invented type.

Nothing in this module decides between those spellings by looking at a single
value - ``1.234`` is genuinely both a grouped thousand and a decimal point. The
value-level parser records which reading it used; :mod:`dtypeinfer.ambiguity`
resolves the column.
"""

import ipaddress
import re
import uuid as uuid_module
from calendar import monthrange
from datetime import date, datetime, time

BOOL_TRUE = frozenset({"true", "t", "yes", "y", "on"})
BOOL_FALSE = frozenset({"false", "f", "no", "n", "off"})

NULL_TOKENS = frozenset({
    "", "-", "--", "---", "?", "??", "n/a", "n.a.", "na", "null", "none",
    "nan", "nil", "not available", "no value", "#n/a", "missing",
})

CURRENCY_SYMBOLS = (
    "USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CNY", "BRL", "CHF",
    "MXN", "RUB", "TRY", "ZAR", "SEK", "NOK", "PLN", "kr", "zł", "Rs", "RM",
    "$", "€", "£", "¥", "₹", "₩", "₽", "¢", "₺", "₴", "₦",
)
_CURRENCY_ALT = "|".join(re.escape(token) for token in
                         sorted(CURRENCY_SYMBOLS, key=len, reverse=True))
_CURRENCY_PREFIX = re.compile(r"(?P<symbol>" + _CURRENCY_ALT + r")\s*(?P<sign>[+-])?",
                              re.IGNORECASE)
_CURRENCY_SUFFIX = re.compile(r"\s*(?P<symbol>" + _CURRENCY_ALT + r")\s*\Z",
                              re.IGNORECASE)

# Digits with optional grouping separators on either side of an optional decimal
# part, so `1.234,56`, `1,234.56`, `1234,56` and `1234.56` are all recognised as
# numbers. Which separator means what is a separate question.
NUMBER_SOURCE = (
    r"(?:"
    r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?"
    r"|\d+[.,]\d+"
    r"|\.\d+"
    r"|\d+"
    r")(?:[eE][+-]?\d+)?"
)
_NUMBER = re.compile(r"[+-]?" + NUMBER_SOURCE + r"\Z")
_INTEGER = re.compile(r"[+-]?\d+\Z")
_GROUPED_INTEGER = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+\Z")
_PERCENT = re.compile(r"[+-]?" + NUMBER_SOURCE + r"\s*%\Z")
_PERCENT_NEGATIVE = re.compile(r"\(\s*" + NUMBER_SOURCE + r"\s*%\s*\)\Z")
_EXPONENT = re.compile(r"[eE][+-]?\d+\Z")
# `1 234,56` (SI / French grouping) only reads as one number when the spaces sit
# exactly where a thousands separator would.
_SPACE_GROUPED = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?\Z")

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                   r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")
_EMAIL_LOCAL_ATOM = re.compile(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+\Z")
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\Z")
_URL = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://"
                  r"(?P<authority>[^\s/?#]*)(?:[/?#][^\s]*)?\Z")
_BRACKETED = re.compile(r"\[(?P<host>[^\]]*)\](?P<tail>.*)\Z")
_PORT = re.compile(r":\d*\Z")
_TIME = re.compile(r"(?P<h>2[0-3]|[01]?\d):(?P<m>[0-5]?\d)"
                   r"(?::(?P<s>[0-5]?\d|60)(?:[.,](?P<f>\d{1,9}))?)?"
                   r"\s*(?P<ap>[APap]\.?[Mm]\.?)?"
                   r"(?P<tz>[Zz]|[Uu][Tt][Cc]|[Gg][Mm][Tt]|[+-]\d{2}:?\d{2})?\Z")
_YEAR_FIRST = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\Z")
_COMPACT = re.compile(r"(\d{4})(\d{2})(\d{2})\Z")
_ORDER_FREE = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2}|\d{4})\Z")
_NAMED_FORMATS = (
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d-%B-%Y",
    "%b %d %Y", "%B %d %Y",
)
# `strptime` is the slowest thing in this module, so it only runs on values that
# could actually be spelled with a month name.
_HAS_LETTER = re.compile(r"[A-Za-z]")


def as_integer(text, decimal_style="auto"):
    """`(int, tag)` only when the value is *written* as a whole number.

    `1.000` is a decimal that happens to be integral, so it stays a float, while
    `1.234` written by someone who groups with dots is an integer. The style
    decides which of those two the dot means.
    """
    body = _clean(text)
    if body is None:
        return None
    if _INTEGER.match(body):
        return int(body), _leading_zero_tag(body)
    if _GROUPED_INTEGER.match(body):
        return int(body.replace(",", "")), "thousands_separator"
    parsed = _number_full(body, decimal_style)
    if parsed is None or not parsed[2]:
        return None
    return int(parsed[0]), parsed[1]


def as_float(text, decimal_style="auto"):
    """`(float, tag)` for any number spelling.

    `decimal_style="eu"` reads an ambiguous `1,234` as a decimal comma; the
    default keeps the grouping reading unless the value itself decides it.
    """
    body = _clean(text)
    if body is None:
        return None
    parsed = _number_full(body, decimal_style)
    if parsed is None:
        return None
    value, tag, integral = parsed
    if integral:
        tag = "integer_valued" if tag == "plain" else "integer_valued+" + tag
    return value, tag


def as_boolean(text):
    body = _clean(text)
    if body is None:
        return None
    lowered = body.lower()
    if lowered in BOOL_TRUE:
        return True, "true_word"
    if lowered in BOOL_FALSE:
        return False, "false_word"
    return None


def as_percent(text, decimal_style="auto"):
    body = _clean(text)
    if body is None:
        return None
    accounting = bool(_PERCENT_NEGATIVE.match(body))
    if not accounting and not _PERCENT.match(body):
        return None
    inner = body.strip("()% ")
    parsed = _number(inner, decimal_style)
    if parsed is None:
        return None
    value, tag = parsed
    if accounting:
        value, tag = -abs(value), "accounting_negative+" + tag
    return value, "percent" if tag == "plain" else "percent+" + tag


def as_currency(text, decimal_style="auto"):
    """`(float, tag)` when a currency token wraps the number, either side."""
    body = _clean(text)
    if body is None:
        return None
    prefix = _CURRENCY_PREFIX.match(body)
    if prefix and _number_ok(body[prefix.end():], decimal_style):
        symbol = prefix.group("symbol")
        number = (prefix.group("sign") or "") + body[prefix.end():]
    else:
        suffix = _CURRENCY_SUFFIX.search(body)
        if suffix is None:
            return None
        symbol = suffix.group("symbol")
        number = body[:suffix.start()]
    parsed = _number(number.strip(), decimal_style)
    if parsed is None:
        return None
    value, tag = parsed
    return value, symbol.upper() + ":" + tag


def _number_ok(text, decimal_style):
    return _number(text.strip(), decimal_style) is not None


def as_date(text):
    """`(date, tag)` - the tag says whether the field order was decidable.

    `31/12/2020` can only be day-first, `12/25/2020` can only be month-first,
    `03/04/2020` is both. Only the first two are safe to coerce silently.
    """
    body = _clean(text)
    if body is None:
        return None
    match = _YEAR_FIRST.match(body)
    if match:
        return _build(match, "iso")
    match = _COMPACT.match(body)
    if match:
        return _build(match, "iso_compact")
    match = _ORDER_FREE.match(body)
    if match:
        return _order_free(match)
    if _HAS_LETTER.search(body):
        for fmt in _NAMED_FORMATS:
            try:
                return datetime.strptime(body, fmt).date(), "named_month"
            except ValueError:
                continue
    return None


def as_time(text):
    body = _clean(text)
    if body is None:
        return None
    match = _TIME.match(body)
    if match is None:
        return None
    hour = int(match.group("h"))
    meridiem = match.group("ap")
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        if meridiem.lower().replace(".", "") == "am":
            hour = 0 if hour == 12 else hour
        elif hour != 12:
            hour += 12
    tags = ["hms" if match.group("s") else "hm"]
    micro = 0
    if match.group("f"):
        micro = int(match.group("f")[:6].ljust(6, "0"))
        tags.append("subsecond")
    if meridiem:
        tags.append("meridiem")
    if match.group("tz"):
        tags.append("timezone_offset_dropped")
    # A leap second appears in real feeds and has nowhere to go in `time`.
    seconds = min(int(match.group("s") or 0), 59)
    return time(hour, int(match.group("m")), seconds, micro), "+".join(tags)


def as_datetime(text):
    """`(datetime, tag)` - a date and a time with a separator between them."""
    body = _clean(text)
    if body is None:
        return None
    # Without a colon there is no clock part, so no split point can ever match.
    # This keeps a plain text column from paying for the scan below.
    if ":" not in body:
        return None
    for index, char in enumerate(body):
        if char not in "Tt ":
            continue
        parsed_date = as_date(body[:index])
        parsed_time = as_time(body[index + 1:])
        if parsed_date is None or parsed_time is None:
            continue
        day, date_tag = parsed_date
        clock, time_tag = parsed_time
        tags = [tag for tag in (date_tag, time_tag) if tag != "iso"]
        return datetime(day.year, day.month, day.day, clock.hour, clock.minute,
                        clock.second, clock.microsecond), "+".join(tags) or "iso"
    return None


def as_uuid(text):
    body = _clean(text)
    if body is None or not _UUID.match(body):
        return None
    return uuid_module.UUID(body.lower()), "canonical"


def as_email(text):
    body = _clean(text)
    if body is None or body.count("@") != 1 or len(body) > 254:
        return None
    local, domain = body.split("@")
    if not 0 < len(local) <= 64:
        return None
    if local.startswith('"'):
        if not local.endswith('"') or len(local) < 3:
            return None
    else:
        if local.startswith(".") or local.endswith(".") or ".." in local:
            return None
        if not all(_EMAIL_LOCAL_ATOM.match(atom) for atom in local.split(".")):
            return None
    if not _valid_domain(domain):
        return None
    return body, "domain_labels_%d" % domain.count(".")


def as_url(text):
    body = _clean(text)
    if body is None:
        return None
    match = _URL.match(body)
    if match is None:
        return None
    _userinfo, seen_at, hostport = match.group("authority").rpartition("@")
    bracketed = _BRACKETED.match(hostport)
    if bracketed:
        if as_ipv6(bracketed.group("host")) is None:
            return None
        if bracketed.group("tail") and not _PORT.match(bracketed.group("tail")):
            return None
    else:
        host, _, port = hostport.partition(":")
        if port and (not port.isdigit() or not 0 < int(port) < 65536):
            return None
        if not (host == "localhost" or as_ipv4(host) or _valid_domain(host)):
            return None
    scheme = match.group("scheme").lower()
    return body, scheme + ("_user_info" if seen_at else "")


def as_ipv4(text):
    body = _clean(text)
    if body is None:
        return None
    try:
        return ipaddress.IPv4Address(body), "single"
    except ValueError:
        return None


def looks_like_ipv4(text):
    """Four number groups, ignoring that `01.2.3.4` is not a valid address."""
    body = _clean(text)
    if body is None or body.count(".") != 3:
        return False
    return all(part.isdigit() and int(part) < 1000 for part in body.split("."))


def as_ipv6(text):
    body = _clean(text)
    if body is None or body.count(":") < 2:
        return None
    try:
        return ipaddress.IPv6Address(body), "single"
    except ValueError:
        return None


def as_null(text, extra=()):
    """Whether a cell is a missing value. `extra` must already be lower-case."""
    body = _clean(text)
    if body is None:
        return True
    lowered = body.lower()
    return lowered in NULL_TOKENS or lowered in extra


def normalize(value):
    """Cells arrive as None, numbers or text; only text can be type-checked."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _number(body, decimal_style="auto"):
    parsed = _number_full(body, decimal_style)
    return None if parsed is None else parsed[:2]


def _number_full(body, decimal_style="auto"):
    """`(float, tag, written_as_whole_number)` for any accepted number spelling."""
    if _SPACE_GROUPED.match(body):
        body = body.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if _NUMBER.match(body) is None:
        return None
    negative = body.startswith("-")
    unsigned = body.lstrip("+-")
    exponent = 0
    exponent_match = _EXPONENT.search(unsigned)
    if exponent_match:
        exponent = int(exponent_match.group(0)[1:])
        unsigned = unsigned[:exponent_match.start()]
    mantissa, kind = _mantissa(unsigned, decimal_style)
    if mantissa is None:
        return None
    value = float(mantissa) * (10 ** exponent)
    if negative:
        value = -value
    tags = []
    if exponent_match:
        tags.append("scientific")
    if kind != "plain":
        tags.append(kind)
    integral = exponent_match is None and "." not in mantissa
    return value, "+".join(tags) or "plain", integral


def _mantissa(unsigned, decimal_style):
    """Rewrite `unsigned` as a plain dot-decimal string, or (None, _).

    Two separators of the same kind can only be grouping marks, and a tail that
    is not three digits long can only be a decimal. Everything else - a lone
    `1.234` - is decided by the column's style, defaulting to the convention the
    separator character itself suggests.
    """
    if not unsigned:
        return None, "plain"
    positions = [index for index, char in enumerate(unsigned) if char in ".,"]
    if not positions:
        return unsigned, "plain"
    last = positions[-1]
    tail_length = len(unsigned) - last - 1
    marks = {unsigned[index] for index in positions}
    groups_here = tail_length == 3 and last > 0 and (
        (len(positions) > 1 and len(marks) == 1)
        or not _is_decimal_mark(unsigned[last], decimal_style))
    if groups_here:
        grouped = _digroup(unsigned, positions)
        if grouped is None:
            return None, "plain"
        return grouped, "grouped_thousands"
    head = unsigned[:last]
    fraction = unsigned[last + 1:]
    head_positions = [index for index in positions if index < last]
    if head_positions:
        stripped = _digroup(head, head_positions)
        if stripped is None:
            return None, "plain"
        head = stripped
    tags = []
    if unsigned[last] == ",":
        tags.append("decimal_comma")
    if head_positions:
        tags.append("grouped_thousands")
    return (head + "." + fraction) if head else ("0." + fraction), "+".join(tags) or "plain"


def _is_decimal_mark(char, decimal_style):
    """Whether `char` is this column's decimal separator, style permitting."""
    if decimal_style == "eu":
        return char == ","
    return char == "."


def _digroup(text, positions):
    """Strip grouping separators after checking each group is 3 digits."""
    if not positions:
        return text
    bounds = positions + [len(text)]
    if bounds[0] > 3 or bounds[0] < 1:
        return None
    for index in range(1, len(positions) + 1):
        if bounds[index] - bounds[index - 1] - 1 != 3:
            return None
    return "".join(text[i:j] for i, j in zip([0] + [b + 1 for b in positions], bounds))


def _clean(text):
    if text is None:
        return None
    if not isinstance(text, str):
        text = normalize(text)
        if text is None:
            return None
    stripped = text.strip()
    return stripped or None


clean = _clean


def _build(match, tag):
    year, month, day = (int(group) for group in match.groups())
    built = _make_date(year, month, day)
    return None if built is None else (built, tag)


def _order_free(match):
    first, second = int(match.group(1)), int(match.group(2))
    raw_year = match.group(3)
    year = int(raw_year)
    two_digit = len(raw_year) == 2
    if two_digit:
        year = year + 2000 if year < 69 else year + 1900
    if first > 12 >= second:
        day, month, order = first, second, "day_first"
    elif second > 12 >= first:
        day, month, order = second, first, "month_first"
    elif first <= 12 and second <= 12:
        day, month = first, second
        order = "order_ambiguous" if first != second else "order_symmetric"
    else:
        return None
    built = _make_date(year, month, day)
    if built is None:
        return None
    if two_digit:
        order = "two_digit_year+" + order
    return built, order


def _make_date(year, month, day):
    if not 1 <= month <= 12 or year < 1000:
        return None
    if not 1 <= day <= monthrange(year, month)[1]:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _leading_zero_tag(body):
    digits = body.lstrip("+-")
    if len(digits) > 1 and digits[0] == "0":
        return "leading_zero"
    return "plain"


def _valid_domain(domain):
    if not 0 < len(domain) <= 253 or domain.startswith((".", "-")) \
            or domain.endswith((".", "-")):
        return False
    labels = domain.split(".")
    if len(labels) < 2 or not all(_LABEL.match(label) for label in labels):
        return False
    top = labels[-1]
    return top.isalpha() and 2 <= len(top) <= 63
