# Predictors of progress in language and reading skills for children with Down syndrome

> [!WARNING]
> This is work in progress. All data and models are preliminary.

This repository hosts an exploratory study of factors associated with language and reading outcomes for children with Down syndrome.

## Getting started

### Clone repositories

In the same directory (perhaps `dseinternational`):

```bash
git clone https://github.com/dseinternational/language-reading-predictors.git
```

For now, also:

```bash
git clone https://github.com/dseinternational/research.git
```

### Prerequisites

#### Fitting models

To fit models, Python 3.14 or later is required.

Then, to create a virtual environment and install Python dependencies, from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

#### Creating reports

To update or create reports, [Quarto](https://quarto.org/docs/get-started/) is required. We also use CSpell for checking spelling, for which a recent installation of [Node.js](https://nodejs.org/en) is required.

To install Node dependencies, from the repository root:

```bash
npm install
```

## License

All source code in this repository is licensed under the GNU Affero General Public License v3.0 **(AGPL-3.0-only)**. See `LICENSE`.

Some other artifacts are licensed under other licenses:

- **Code**: GNU Affero General Public License v3.0 (AGPL-3.0) — see `LICENSE`.
- **Documentation, reports and papers**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `docs/LICENSE`.
- **Data**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `data/LICENSE` for details.

AGPL-3.0 requires that if you modify and run this software to provide a network service, you must offer the corresponding source code to users of that service.
