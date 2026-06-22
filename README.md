# Predictors of progress in language and reading skills for children with Down syndrome

> [!WARNING]
> This is work in progress. All data and models are preliminary.

**This repository hosts an exploratory study of factors associated with language and reading outcomes for children with Down syndrome.**

## About this study

All children with Down syndrome experience delays in language and reading development. If we can identify the factors that contribute to better outcomes, then we may be able to offer families and practitioners better advice about effective teaching strategies and interventions.

This project draws together data collected on a variety of language, reading and other measures in studies of children with Down syndrome. We are using a variety of machine learning and modern statistical techniques to describe what these data show, and to estimate the influences of multiple factors on rates of language and literacy learning.

The first data set that we are exploring is from a longitudinal study of children with Down syndrome who took part in the [Reading and Language Intervention (RLI) trial](https://www.down-syndrome.org/resources/reading-language-intervention/), which followed 54 children across four timepoints. The RCT component of this study was [previously reported](https://doi.org/10.1111/j.1469-7610.2012.02557.x), as were analyses of [speech production accuracy](https://doi.org/10.1111/jir.12890) and an associated [investigation of teaching blending skills](https://doi.org/10.1177/0265659012474674).

With this data set, we are using gradient boosting (machine learning algorithms that combine multiple decision trees) to train models that predict gains and achievement levels from the available variables. We then analyse these trained models to understand which variables contribute to the best predictions. This offers a data-driven approach to identifying predictors that may be important for different outcomes. The second phase of our exploration takes these candidate predictors and develops statistical models to estimate the independent and joint effects of selected predictors on outcomes of interest. We use Bayesian inference to obtain full posterior probability distributions for all parameters in our models in order to quantify uncertainty in our estimates.

### Open and reproducible

We are developing, evaluating and iterating our models openly in this repository, where we share all source code and anonymised source data under open licenses.

### Future directions

Over time, we may extend this project to explore further data sets and modelling techniques and welcome input from interested partners on how the project might evolve.

When Down Syndrome Education International launches its [LearningTracker app and services](https://www.learningtracker.app/) we expect to make additional parent-reported vocabulary data available to extend this project in the future.

## Contributing

We welcome partners interested in developing statistical models, evaluating and interpreting findings, and sharing original data.

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

To fit models, a recent Python installation is required. Some of our dependencies are best installed from [conda-forge](https://conda-forge.org/), for which either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Miniforge](https://conda-forge.org/download/) is required.

Then, to install Python dependencies, from the repository root:

```bash
conda env update -f environment.yml
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
