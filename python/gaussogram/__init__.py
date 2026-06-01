"""gaussogram — a parsimonious adaptive time-frequency representation
(fast Gaussian S-transform / GFT)."""

from ._api import (
    BandLayout,
    Gaussogram1d,
    gft1d,
    gft1d_real,
    inverse_real,
    output_len,
    partitions,
    real_partitions,
    scheme_bands,
    to_grid,
)

__all__ = [
    "BandLayout",
    "Gaussogram1d",
    "gft1d",
    "gft1d_real",
    "inverse_real",
    "output_len",
    "partitions",
    "real_partitions",
    "scheme_bands",
    "to_grid",
]

__version__ = "0.1.0"
