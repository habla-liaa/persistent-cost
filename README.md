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

## Development


## Authors

- Manuela Cerdeiro
- Pablo Riera
- Thiago Tiracha
- Francisco Gozzi

## References


