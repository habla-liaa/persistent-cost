# Persistent Cost

A Python library for computing persistent homology cost and related topological data analysis tools.

## Features

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install from source

1. Clone the repository:
```bash
git clone https://github.com/habla-liaa/persistent-cost.git
cd persistent-cost
```

2. Install in editable mode (recommended for development):
```bash
pip install -e .
```

This will automatically install all required dependencies:
- numpy
- scipy
- pandas
- matplotlib
- gudhi

### Install with development dependencies

To install additional testing tools:
```bash
pip install -e ".[dev]"
```

### Enable accelerated pivot backends

The fast Numba and Cython versions of the pivot algorithm are packaged as optional extras:

```bash
pip install -e ".[accel]"
```

After installing the extra, build the Cython extension in-place (required once per environment):

```bash
python setup.py build_ext --inplace
```

The accelerated variants live in `persistent_cost.utils.algorithms_fast` and expose helpers such as
`do_pivot_fast`, `do_pivot_numba`, and `do_pivot_cython`.



## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test file

```bash
pytest tests/test_utils.py -v
```

### Run tests with coverage

```bash
pytest tests/ --cov=persistent_cost --cov-report=html
```

### Run tests directly

```bash
python tests/test_utils.py
```

### Benchmark the pivot implementations

Install the development extras to pull in the Fire CLI helper, ensure at least one accelerated
backend is available, then compare runtimes:

```bash
pip install -e ".[dev]"
python -m benchmarks.benchmark_pivot benchmark --points=10,30,50 --repeats=5
```

Use `--progress=False` to disable progress bars or override any of the other parameters, e.g.
`--threshold=0.4 --maxdim=3`.

## Development


## Authors

- Manuela Cerdeiro
- Pablo Riera
- Thiago Tiracha
- Francisco Gozzi

## References


