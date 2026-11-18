> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Dependency update, 18 November 2026

The Python lockfile updates 17 packages, including NumPy 2.5.5, Optuna 5.1.0, its integration package and Ruff 0.17.5. The development minimums now match the locked versions of mypy and Ruff. NumPy changes how it handles casts from structured arrays to its string type. Optuna drops Python 3.9 support. This project uses Python 3.14. [NumPy release notes](https://github.com/numpy/numpy/releases/tag/v2.5.5), [Optuna release notes](https://github.com/optuna/optuna/releases/tag/v5.1.0).

The Node tools update to pnpm 12.21.0, CSpell 10.5.1 and Prettier 3.10.1. The shared theme pin moves to `tokens/2026.3.8`. All 11 copied theme, font and licence files match that release byte for byte. The shared research library and the four GitHub Actions are already at their latest stable releases.

This update uses the same one-day release cutoff as the 11 November update. New releases of CSpell and its packages, `fast-equals`, `cloudpickle`, `filelock`, `platformdirs` and `soupsieve` fall inside that window. PyTensor 3.3.8 caps Numba at 0.68.0, PyMC 6.4.0 requires Cachetools below 7, and CSpell's `global-directory` dependency pins `ini` to 6.0.0. Both PyMC and PyTensor require Python below 3.15. [Previous update](20261111-dependency-update.md).
