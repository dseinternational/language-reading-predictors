> [!NOTE]
> Conciseness edits by a LLM-based AI tool (Codex/GPT-6).

# Predictors of progress in language and reading skills for children with Down syndrome

> [!WARNING]
> This is work in progress. All data and models are preliminary.

This repository contains an exploratory study of factors associated with language and reading outcomes in children with Down syndrome.

## About this study

Children with Down syndrome experience delays in language and reading development. Understanding the factors associated with better outcomes may help families and practitioners choose teaching strategies and interventions.

We analyse language, reading and related measures from the [Reading and Language Intervention (RLI) trial](https://www.down-syndrome.org/resources/reading-language-intervention/). Earlier publications report the [randomised trial](https://doi.org/10.1111/j.1469-7610.2012.02557.x), [speech production accuracy](https://doi.org/10.1111/jir.12890) and [teaching of blending skills](https://doi.org/10.1177/0265659012474674).

We first use gradient boosting, which combines decision trees, to identify variables that help predict gains and achievement levels. Bayesian models then estimate treatment contrasts and adjusted associations, with probability distributions that describe their uncertainty. [METHODS.md](METHODS.md) explains the methods and limits on causal interpretation.

## Documentation

Start with the [documentation guide](docs/README.md). It links the methods, model catalogue, worked example and refit instructions. The [notes guide](notes/README.md) separates dated findings from decisions that still govern the analysis. The integrated report is a draft.

## Contributing

We share source code and anonymised data under open licences. We welcome partners to develop models, interpret findings, contribute data and explore further datasets or methods.

## Getting started

### Clone the repository

```bash
git clone https://github.com/dseinternational/language-reading-predictors.git
cd language-reading-predictors
```

`uv sync` fetches the pinned `dse-research-utils` dependency from GitHub. A sibling `research` checkout is needed only for local development of that library; see `pyproject.toml` for the path-source option.

### Prerequisites

For a worked introduction to the Bayesian code, follow [one statistical model from synthetic data to a reported difference](docs/learning/itt-model-walkthrough.md).

#### Fitting models

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which also provides Python. From the repository root, create the environment:

```bash
uv sync
```

Run commands with `uv run`, for example `uv run pytest` or `uv run python scripts/fit_model.py lrp-rli-gbg-001`. Activating `.venv` also works.

Supported platforms are Linux (x86-64 and arm64), Apple Silicon macOS, and Windows (x86-64). Intel macOS is not supported, because [numba](https://numba.pydata.org/) no longer publishes macOS x86-64 wheels.

Plotting model graphs additionally requires the system [Graphviz](https://graphviz.org/) `dot` binary, which is not a Python package: `brew install graphviz`, `apt install graphviz` or `winget install Graphviz.Graphviz`.

#### Creating reports

Install [Quarto](https://quarto.org/docs/get-started/) to create reports and [Node.js](https://nodejs.org/en) to run the spelling and formatting tools.

Install the Node dependencies from the repository root:

```bash
npm install
```

## License

- Code uses the GNU Affero General Public License, version 3 or later (`AGPL-3.0-or-later`). See the [package metadata](pyproject.toml) and source-file headers.
- Documentation, reports and papers use Creative Commons Attribution 4.0 International (`CC BY 4.0`). See [docs/LICENSE](docs/LICENSE).
- Data use Creative Commons Attribution 4.0 International (`CC BY 4.0`). See [data/LICENSE](data/LICENSE).
