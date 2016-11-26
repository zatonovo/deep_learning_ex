DOCKER?=docker
PYTHON?=python
.PHONY: all run bash python test_openface
PROJECT=$(shell basename `pwd`)
HOST_DIR = $(shell pwd)
IMAGE_NAME = dlx/${PROJECT}


all:
	${DOCKER} build -t ${IMAGE_NAME} .

torch:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

bash:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} bash

python:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} ${PYTHON}

