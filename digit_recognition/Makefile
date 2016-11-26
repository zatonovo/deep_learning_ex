PROC?=cpu
DOCKER?=docker
PYTHON?=python
THEANO_FLAGS?=
.PHONY: all run torch bash python test_openface
PROJECT=$(shell basename `pwd`)
HOST_DIR = $(shell pwd)
IMAGE_NAME = zatonovo/dlx_${PROC}:latest


torch:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

bash:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} bash

python:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code -e ${IMAGE_NAME} ${PYTHON}

