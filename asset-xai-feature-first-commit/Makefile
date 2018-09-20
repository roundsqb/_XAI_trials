install:
	pip install .

test:
	pytest --cov xai --cov-report=term-missing tests

lint:
	pylint --disable=duplicate-code xai
	pylint --disable=missing-docstring,redefined-outer-name,duplicate-code tests

package:
	python setup.py bdist_wheel upload -r pypi-qb
	python setup.py bdist_egg upload -r pypi-qb
