PYTHON?=python
.PHONY: all run bash python test_openface
PROJECT=$(shell basename `pwd`)
HOST_DIR = $(shell pwd)
IMAGE_NAME = dlx/${PROJECT}


all:
	docker build -t ${IMAGE_NAME} .

torch:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

bash:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} bash

python:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} python

