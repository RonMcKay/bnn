.PHONY: setup install pre-commit test clean

setup: install pre-commit

install:
	@echo "Installing dependencies..."
	poetry install

pre-commit: install
	@echo "Setting up pre-commit..."
	poetry run pre-commit install -t commit-msg -t pre-commit

test: test-flake8 test-black test-isort

test-flake8:
	@echo "Checking format with flake8..."
	poetry run flake8 . --count --statistics

test-black:
	@echo "Checking format with black..."
	poetry run black --check .

test-isort:
	@echo "Checking format with isort..."
	poetry run isort --check --settings-path pyproject.toml .

format:
	@echo "Formatting with black and isort..."
	poetry run black
	poetry run isort --settings-path pyproject.toml

clean:
	rm -rf build sdist __pycache__
