PROC?=cpu
DOCKER?=docker
PYTHON?=python
THEANO_FLAGS?=
.PHONY: all gpu run bash python test_openface
PROJECT=$(shell basename `pwd`)
HOST_DIR = $(shell pwd)
CPU_IMAGE = zatonovo/dlx_cpu:latest
GPU_IMAGE = zatonovo/dlx_gpu:latest
IMAGE_NAME = zatonovo/dlx_${PROC}:latest


all: cpu gpu

cpu:
	${DOCKER} build -t ${CPU_IMAGE} -f Dockerfile-cpu .
gpu:
	${DOCKER} build -t ${GPU_IMAGE} -f Dockerfile-gpu .

torch:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} th

bash:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

python:
	${DOCKER} run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} ${PYTHON}

