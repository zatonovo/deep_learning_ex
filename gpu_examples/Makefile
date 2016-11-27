DOCKER?=nvidia-docker
PYTHON?=python
THEANO_FLAGS?=
.PHONY: all run bash python test_openface
PROJECT=$(shell basename `pwd`)
HOST_DIR = $(shell pwd)
IMAGE_NAME = zatonovo/dlx_gpu:latest


all: bash


torch:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} th

bash:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

python:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code -e ${IMAGE_NAME} ${PYTHON}

