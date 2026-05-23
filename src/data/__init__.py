"""Data loading and preprocessing for prism-vrec.

Importing this package as a whole has two side effects:

1. :mod:`src.data.dvbpr` registers the four DVBPR datasets in the
   provider registry exposed by :mod:`src.data.base`.
2. :mod:`src.data.auto_register` scans ``plugins/datasets/`` and
   registers any user-provided dataset found there (see that module's
   docstring for the directory layout).
"""

from src.data.base import (
    DatasetProvider,
    get_dataset_provider,
    register_dataset_provider,
    registered_dataset_names,
    validate_layout,
    write_processed_splits,
)
from src.data.dvbpr import DVBPRDataLoader
from src.data.example_csv import CSVDatasetProvider
from src.data.synthetic import SyntheticDatasetProvider
from src.data.preprocessing import (
    build_mappings,
    kcore_filter,
    leave_one_out_split,
)

from src.data import auto_register  # noqa: F401

__all__ = [
    "CSVDatasetProvider",
    "DatasetProvider",
    "DVBPRDataLoader",
    "SyntheticDatasetProvider",
    "build_mappings",
    "get_dataset_provider",
    "kcore_filter",
    "leave_one_out_split",
    "register_dataset_provider",
    "registered_dataset_names",
    "validate_layout",
    "write_processed_splits",
]
