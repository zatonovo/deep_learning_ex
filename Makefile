PYTHON?=python
.PHONY: all run bash python
HOST_DIR = ~/workspace/deep_learning_ex
IMAGE_NAME = zatonovo/dlx


all:
	docker build -t ${IMAGE_NAME} .

torch:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

bash:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} bash

python:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} python
