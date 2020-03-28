FLAGS=


black:
	black time_series_experiments tests setup.py


flake:
	flake8 time_series_experiments tests setup.py


test: flake
	py.test -s -vv $(FLAGS) ./tests/


mypy:
	mypy time_series_experiments --ignore-missing-imports --disallow-untyped-calls --no-site-packages --strict


cov cover coverage: flake
	py.test -s -v --cov-report term --cov-report html --cov time_series_experiments ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


cov_only: flake
	py.test -s -v --cov-report term --cov-report html --cov time_series_experiments ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


install:
	pip install -r requirements.txt


develop:
	python setup.py develop


build:
	python setup.py build


clean:
	rm -rf `find . -name __pycache__`
	rm -rf `find . -name .pytest_cache`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '@*' `
	rm -f `find . -type f -name '#*#' `
	rm -f `find . -type f -name '*.orig' `
	rm -f `find . -type f -name '*.rej' `
	rm -f .coverage
	rm -rf coverage
	rm -rf build
	rm -rf htmlcov
	rm -rf dist
	rm -rf bin/*
	rm -rf *.egg-info


doc:
	make -C docs html
	@echo "open file://`pwd`/docs/_build/html/index.html"


.PHONY: all flake test cov clean doc install