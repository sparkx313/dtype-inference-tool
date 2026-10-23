"""dtypeinfer - work out what a column of text actually holds.

    from dtypeinfer import infer_column, infer_csv

    infer_column(["2026-01-31", "2026-02-01", "2026-02-28"]).type
    # 'date'

The type is only half of the answer. Each result also carries the fit of every
candidate type, the spellings found (`decimal_comma`, `named_month`,
`leading_zero`) and the ambiguities a reviewer has to decide (`order_ambiguous`,
`ambiguous_three_digit_tail`, `also_fits`).
"""

from .ambiguity import resolve_decimal_style
from .coerce import CoercionReport, coerce_column, coerce_value
from .csvread import infer_csv
from .infer import ColumnInference, TableInference, infer_column, infer_table
from .sample import ColumnSampler, reservoir_sample
from .types import PYTHON_TYPES, SPECIFICITY, Settings

__version__ = "0.1.0"

__all__ = [
    "ColumnInference",
    "ColumnSampler",
    "CoercionReport",
    "PYTHON_TYPES",
    "SPECIFICITY",
    "Settings",
    "TableInference",
    "coerce_column",
    "coerce_value",
    "infer_column",
    "infer_csv",
    "infer_table",
    "resolve_decimal_style",
    "reservoir_sample",
]
