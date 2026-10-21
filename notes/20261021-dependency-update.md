> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Dependency update, 21 October 2026

The lock files update nine Python packages and the CSpell npm dictionary. The shared research library, direct Node tools and CI actions already use their latest stable releases. The Python requirement, supported platforms and source of the shared library remain unchanged.

| Package          | Previous version | Updated version |
| ---------------- | ---------------- | --------------- |
| certifi          | 2026.7.22        | 2026.10.19      |
| duckdb           | 1.5.6            | 2.0.0           |
| filelock         | 4.2.0            | 4.2.4           |
| jsonpointer      | 3.2.1            | 3.4             |
| mistune          | 3.3.4            | 3.4.0           |
| numcodecs        | 0.17.0           | 0.17.1          |
| platformdirs     | 4.13.3           | 4.13.5          |
| setuptools       | 84.0.0           | 84.2.1          |
| wcwidth          | 0.9.2            | 0.9.3           |
| @cspell/dict-npm | 5.2.54           | 5.2.55          |

DuckDB arrives through the shared library's `columnar` extra. Searches found no DuckDB imports or calls in the repository's Python files or the installed shared library's Python files. Two notebook environment banners list its version. We retain the major update because we found no database workload here that needs migration.

DuckDB's [migration guide](https://www.duckdb.org/docs/preview/operations_manual/migration) states that its new default database file format cannot be read by older releases. A manual compatibility check also exposed changes to its Python API. Relations no longer have `fetchone()`, and a partly consumed `execute()` result prevents further queries on the same connection. The final check used `sql(...).fetchall()` and `read_parquet(...).aggregate(...).fetchall()`. Both returned the expected totals from a small pandas table and a temporary Parquet file.

Manual checks also passed for a Numcodecs compression round trip, notebook HTML export through nbconvert and JSON pointer resolution. Both lock files passed their install checks, and the installed Python packages passed `uv pip check`. Validation does not cover external database files or SQL scripts. No scientific model fits were rerun.
