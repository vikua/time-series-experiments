FLAGS=


black:
	black transformer tests setup.py


flake:
	flake8 transformer tests setup.py


test: flake
	py.test -s -v $(FLAGS) ./tests/


mypy:
	mypy transformer --ignore-missing-imports --disallow-untyped-calls --no-site-packages --strict


cov cover coverage: flake
	py.test -s -v --cov-report term --cov-report html --cov transformer ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


cov_only: flake
	py.test -s -v --cov-report term --cov-report html --cov transformer ./tests
	@echo "open file://`pwd`/htmlcov/index.html"


install:
	pip install -r requirements-dev.txt


clean:
	rm -rf `find . -name __pycache__`
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


doc:
	make -C docs html
	@echo "open file://`pwd`/docs/_build/html/index.html"


.PHONY: all flake test cov clean doc install