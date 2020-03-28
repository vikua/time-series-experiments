import os
import re
import sys
from setuptools import setup, find_packages


PY_VER = sys.version_info

if not PY_VER >= (3, 6):
    raise RuntimeError("Sorry, only Python 3.6 and later")


def read(f):
    return open(os.path.join(os.path.dirname(__file__), f)).read().strip()


def read_requirements(fname):
    with open(fname) as requirements_file:
        reqs = requirements_file.read().split("\n")
        requirements_w_version = [r.split(";") for r in reqs]
        reqs = [r[0] for r in requirements_w_version if len(r) == 1 or ">" in r[1]]
        return [
            x
            for x in reqs
            if x.strip() != "" and not x.startswith("-") and not x.startswith("#")
        ]


requirements = read_requirements("requirements.txt")
extras_require = {}


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), "time_series_experiments", "__init__.py"
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            msg = "Cannot find version in time_series_experiments/__init__.py"
            raise RuntimeError(msg)


classifiers = [
    "License :: MIT License",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Operating System :: POSIX",
    "Development Status :: 2 - Pre-Alpha",
]


setup(
    name="time_series_experiments",
    version=read_version(),
    description=("time series experiments"),
    long_description=read("README.md"),
    classifiers=classifiers,
    platforms=["POSIX"],
    author="Viktor Kovryzhkin",
    author_email="vik.kovrizhkin@gmail.com",
    url="https://github.com/vikua/time-series-experiments",
    download_url="",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    tests_require=requirements,
    extras_require=extras_require,
    keywords=[
        "machine learning",
        "deep learning",
        "time series analysis",
        "time series",
    ],
    zip_safe=True,
    include_package_data=True,
)
