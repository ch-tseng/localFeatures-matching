#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import cv2
import imutils
import argparse

def ROOTSIFT(grayIMG, kpsData):
    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kps, descs) = extractor.compute(grayIMG, kpsData)

    if len(kps) > 0:
        #L1-正規化
        eps=1e-7
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        #取平方根
        descs = np.sqrt(descs)

        return (kps, descs)
    else:
        return ([], None)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="Path to first image")
ap.add_argument("-s", "--second", required=True, help="Path to second image")
args = vars(ap.parse_args())

detector = cv2.FeatureDetector_create("SURF")
matcher = cv2.DescriptorMatcher_create("BruteForce")

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width = 600)
imageB = imutils.resize(imageB, width = 600)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

kpsA = detector.detect(grayA)
kpsB = detector.detect(grayB)

(kpsA, featuresA) = ROOTSIFT(grayA, kpsA)
(kpsB, featuresB) = ROOTSIFT(grayB, kpsB)

rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

matches = []

for m in rawMatches:
    print ("#1:{} , #2:{}".format(m[0].distance, m[1].distance))
    if len(m) == 2 and m[0].distance < m[1].distance * 0.7:
        matches.append((m[0].trainIdx, m[0].queryIdx))

print("# of keypoints from first image: {}".format(len(kpsA)))
print("# of keypoints from second image: {}".format(len(kpsB)))
print("# of matched keypoints: {}".format(len(matches)))

(hA, wA) = imageA.shape[:2]
(hB, wB) = imageB.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = imageA
vis[0:hB, wA:] = imageB

for (trainIdx, queryIdx) in matches:
	color = np.random.randint(0, high=255, size=(3,))
	ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
	ptB = (int(kpsB[trainIdx].pt[0] + wA), int(kpsB[trainIdx].pt[1]))
	cv2.line(vis, ptA, ptB, color, 1)

cv2.imshow("Matched", vis)
cv2.waitKey(0)
