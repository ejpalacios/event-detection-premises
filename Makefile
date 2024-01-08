.PHONY: isort black mypy test local-ci clean docker-build docker-push

IMAGE_NAME := detector
IMAGE_TAG := latest
DOCKER_REGISTRY := docker.io/ejpalacios
PACKAGE := event_detection
TEST := tests

## Create virtual environment
.venv/bin/activate: pyproject.toml
	poetry install

## Export requirements file
requirements.txt: pyproject.toml
	poetry export -o requirements.txt -f requirements.txt --without-hashes

run: .venv/bin/activate
	poetry run python -m $(PACKAGE) --config config.yaml

## Sort imports
isort: .venv/bin/activate
	poetry run isort $(PACKAGE) $(TEST) --check-only

## Check formatting with black
black: .venv/bin/activate
	poetry run black $(PACKAGE) $(TEST) --check

## Mypy static checker
mypy: .venv/bin/activate
	poetry run mypy $(PACKAGE) $(TEST) --install-types --non-interactive

## Run tests
test: .venv/bin/activate
	poetry run pytest --cov=$(PACKAGE) -v

## Run local CI
local-ci: isort black mypy test

docker-build: requirements.txt
	docker build -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG) . 

docker-push: docker-build
	docker login $(DOCKER_REGISTRY)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

## Clean files
clean:
	rm -f requirements.txt
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	rm -rf .venv

