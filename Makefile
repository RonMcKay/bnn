all: test

test:
	poetry run flake8 . --count --statistics
	poetry run black --check .
	poetry run isort --check --settings-path pyproject.toml --recursive .

clean:
	rm -rf wheel sdist __pycache__ __local__storage__
