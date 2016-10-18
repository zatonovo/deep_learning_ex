PROJECT=digit_recognition
PYTHON?=python
.PHONY: all run bash python test_openface
HOST_DIR = ~/workspace/deep_learning_ex/${PROJECT}
IMAGE_NAME = dlx/${PROJECT}


all:
	docker build -t ${IMAGE_NAME} .

torch:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME}

bash:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} bash

python:
	docker run -it --rm -v ${HOST_DIR}:/code ${IMAGE_NAME} python

