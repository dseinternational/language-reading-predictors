<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!--
DRAFT ONLY — not posted. Intended for pymc-devs/pymc as a bug report.
Reviewed and posted by a human; see language-reading-predictors#453.
-->

**Title:** `compute_log_prior` / `compute_log_likelihood` fail for any transform whose name contains an underscore (`LKJCorr`, `LogExpM1`)

---

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5, revised by Claude Code/Fable 5).

### Describe the issue

`get_untransformed_name` recovers a variable's untransformed name by dropping a **fixed three** trailing underscore-separated components — one for the transform name and two for the `__` marker:

```python
# pymc/util.py
def get_transformed_name(name, transform):
    return f"{name}_{transform.name}__"

def get_untransformed_name(name):
    if not is_transformed_name(name):
        raise ValueError(f"{name} does not appear to be a transformed name")
    return "_".join(name.split("_")[:-3])
```

That is only correct when `transform.name` itself contains no underscore. Two shipped transforms violate this:

| Transform class                                     | `name`          | value var              | `get_untransformed_name` |     |
| --------------------------------------------------- | --------------- | ---------------------- | ------------------------ | --- |
| `CholeskyCorrTransform` (the default for `LKJCorr`) | `cholesky_corr` | `corr_cholesky_corr__` | `corr_cholesky`          | ❌  |
| `LogExpM1`                                          | `log_exp_m1`    | `x_log_exp_m1__`       | `x_log_exp`              | ❌  |

Every other shipped transform round-trips correctly (`log`, `logodds`, `ordered`, `sumto1`, `zerosum`, `cholesky-cov`, `cholesky-cov-packed`).

The user-visible consequence is that **`compute_log_prior` and `compute_log_likelihood` cannot be run at all** on any model containing an `LKJCorr` (or `LogExpM1`-transformed) variable. `compute_log_density` builds its elemwise function over `remove_value_transforms(model).value_vars` but subsets the posterior by `[rv.name for rv in model.free_RVs]`; `remove_value_transforms` → `change_value_transforms` renames each value variable through `get_untransformed_name(value.name)`, so the two name sets no longer agree and `xarray` raises.

This blocks any downstream consumer that needs those groups. In our case it is power-scaling prior sensitivity (`arviz_stats.psense`), which requires both `log_prior` and `log_likelihood`, so a model using `LKJCorr` for a correlation matrix cannot be checked for prior sensitivity at all.

### Reproducible code example

A single-variable model with no data is enough:

```python
import pymc as pm

with pm.Model() as m:
    pm.LKJCorr("corr", n=3, eta=2.0)
    idata = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

pm.compute_log_prior(idata, model=m)
```

The same failure with `LogExpM1`:

```python
import pymc as pm
from pymc.distributions import transforms as tr

with pm.Model() as m:
    pm.HalfNormal("x", 1.0, default_transform=tr.log_exp_m1)
    idata = pm.sample(draws=10, tune=10, chains=1, progressbar=False)

pm.compute_log_prior(idata, model=m)
```

And the underlying round-trip, without sampling:

```python
from pymc.util import get_transformed_name, get_untransformed_name

class T:
    def __init__(self, name):
        self.name = name

get_untransformed_name(get_transformed_name("myvar", T("log")))            # 'myvar'          OK
get_untransformed_name(get_transformed_name("myvar", T("cholesky-cov")))   # 'myvar'          OK
get_untransformed_name(get_transformed_name("myvar", T("cholesky_corr")))  # 'myvar_cholesky' BUG
get_untransformed_name(get_transformed_name("myvar", T("log_exp_m1")))     # 'myvar_log_exp'  BUG
```

### Error message

```
ValueError: exact match required for all data variable names, but ['corr_cholesky'] != ['corr']: {'corr', 'corr_cholesky'} are not in both.
```

and for the `LogExpM1` case:

```
ValueError: exact match required for all data variable names, but ['x_log_exp'] != ['x']: {'x_log_exp', 'x'} are not in both.
```

### PyMC version information

```
PyMC: 6.2.0
PyTensor: 3.2.3
ArviZ: 1.2.0
xarray: 2026.7.0
Python: 3.14.6
OS: Darwin 25.5.0 (arm64)
Installed via conda-forge
```

Both reproducers above were re-run on this environment; the two error messages are
quoted from that run. Originally observed on PyMC 6.1.0 / PyTensor 3.1.2, so the
behaviour is unchanged across both releases.

`cholesky_corr` and `log_exp_m1` are the only two shipped transform names containing an underscore — checked exhaustively over `pymc.distributions.transforms` — so they are the only two affected.

### Context for the issue

A possible fix, offered tentatively since I do not know the constraints on `get_untransformed_name` as public API: the one call site that causes this failure does not need to parse the name at all. In `change_value_transforms` (`pymc/model/transform/conditioning.py`) the loop recovers the untransformed name by string surgery on the value variable:

```python
try:
    untransformed_name = get_untransformed_name(value.name)
except ValueError:
    untransformed_name = value.name
```

but the same iteration already has that name directly — a few lines later it passes `node.op.name` (the model variable's untransformed name; `rv.name` from `rv, value = node.inputs` carries it too, verified on 6.1.0) into `model_free_rv` when reconstructing the node. Deriving `untransformed_name` from `node.op.name` instead of parsing `value.name` would fix every underscore-named transform at once and remove the implicit "transform names must not contain underscores" constraint, without touching `get_untransformed_name` itself.

If `get_untransformed_name` should also be fixed for its other callers, note that it cannot be made correct from the string alone in general — a variable genuinely named `myvar_cholesky` transformed by a hypothetical `corr` transform is indistinguishable from `myvar` transformed by `cholesky_corr` — so it would need the transform passed in, or the naming constraint documented and enforced (e.g. a check that `transform.name` contains no underscore, which would at least turn a silent mismatch into a clear error at definition time).
