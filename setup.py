import sys
from pathlib import Path

from setuptools import setup
from setuptools.extension import Extension

try:
    import pybind11
except Exception:
    raise RuntimeError("pybind11 is required to build the extension")


def get_include_dirs():
    root = Path(__file__).parent
    include_dirs = [
        str(root / "include"),
        pybind11.get_include(),
    ]
    return include_dirs


extra_compile_args = []
extra_link_args = []

if sys.platform == "darwin":
    extra_compile_args += ["-std=c++17", "-O3", "-mmacosx-version-min=10.15"]
elif sys.platform.startswith("linux"):
    extra_compile_args += ["-std=c++17", "-O3"]
else:
    extra_compile_args += ["-std=c++17", "-O3"]


extensions = [
    Extension(
        name="abcluster",
        sources=[
            str(Path("bindings") / "pybind_abcluster.cpp"),
        ],
        include_dirs=get_include_dirs(),
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]


setup(
    ext_modules=extensions,
    entry_points={
        "console_scripts": [
            "clonify=clonify.cli:clonify",
        ]
    },
)
