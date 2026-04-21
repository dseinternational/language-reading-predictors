# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import platform
import sys
import arviz as az
import pymc as pm
import pytensor
import torch
import numpy as np
import pandas as pd
import sklearn as skl
import scipy as sp
import joblib
import datetime
import psutil
from typing import Any

RANDOM_SEED = 47


def get_environment_info() -> dict[str, str | bool | Any]:
    mem = psutil.virtual_memory()
    return {
        "date": datetime.datetime.now().isoformat(),
        "platform": platform.platform(),
        "platform_version": platform.version(),
        "cpu": platform.processor(),
        "cores": joblib.cpu_count(),
        "physical_cores": joblib.cpu_count(only_physical_cores=True),
        "ram": f"{mem.total // (1024**3)} GB",
        "ram_available": f"{mem.available // (1024**3)} GB",
        "cuda": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_0": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": sp.__version__,
        "sklearn": skl.__version__,
        "pytorch": torch.__version__,
        "pymc": pm.__version__,
        "pytensor": pytensor.__version__,
        "arviz": az.__version__,
    }


def print_environment_info():
    from dse_research_utils.console.sections import section_header
    from dse_research_utils.console.tables import print_key_value_table

    info = get_environment_info()
    section_header("Environment Information")
    print_key_value_table(info, key_header="Field", value_header="Value")
