from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
import numpy as np
from setuptools import setup


def _extension_modules():
    base = Path("src") / "persistent_cost" / "algorithms"
    sources = [
        base / "_pivot_dense_cython.pyx",
        base / "_pivot_sparse_cython.pyx",
    ]
    directives = {
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
    }
    extensions = cythonize(
        [str(source) for source in sources], compiler_directives=directives)
    
    include_dir = np.get_include()
    for ext in extensions:
        ext.include_dirs.append(include_dir)
    return extensions


setup(ext_modules=_extension_modules())
