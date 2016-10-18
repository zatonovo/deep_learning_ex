#!/usr/bin/env python2
"""
Based off of compare.py
"""
import argparse
import cv2
import itertools
import os
import numpy as np
np.set_printoptions(precision=2)
import openface

HOME = '/root/openface'
IMG_DIM = 96

modelDir = os.path.join(HOME, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(
  os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(
  os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), IMG_DIM)


def align_face(img_path, save=False):
  img_bgr = cv2.imread(img_path)
  if img_bgr is None:
    raise Exception("Unable to load image '%s'" % img_path)

  # Convert to RGB
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  # Get bounding box
  bbox = align.getLargestFaceBoundingBox(img_rgb)
  if bbox is None:
    raise Exception("Unable to find a face in '%s'" % img_path)

  print "Align '%s'" % img_path
  face = align.align(IMG_DIM, img_rgb, bbox,
    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

  if save:
    print "Save to '%s'" % save
    cv2.imwrite(save, face)
  return face


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  [ align_face(p, os.path.basename(p)) for p in args.imgs ]
  pass
