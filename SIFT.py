import cv2
import numpy as np
import argparse
from sklearn import svm
import pickle
from sklearn.externals import joblib


labels = ["Gothic","Korean","Islamic","Georgian"]
#take image path as input
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

#original
path = args["image"]
queryImage = cv2.imread(path)
cv2.imshow("Query Image", queryImage)
print "This is your query Image"
cv2.waitKey(0)

#resize and grayscale
gray = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
gray2 = cv2.resize(gray, (256,256)) 
cv2.imshow("Grayscale resized Image", gray2)
print "1. Image is resized and converted to grayscale"
cv2.waitKey(0)

#gaussian blur
blurredImage = cv2.GaussianBlur(gray2,(11,11),0)
cv2.imshow("Gaussian Blur", blurredImage)
print "2. Gaussian Blur is applied"
cv2.waitKey(0)

#sift keypoints
sift = cv2.xfeatures2d.SIFT_create()
kp ,dsc = sift.detectAndCompute(blurredImage,None)
# print kp
img = cv2.drawKeypoints(blurredImage, kp, gray2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg', img)
cv2.imshow('SIFT', img)
print "3. Extraction of features using SIFT algorithm"
cv2.waitKey(0)

#prediction
voc = np.genfromtxt("423VOCAB.csv",delimiter = " ",dtype = "float32")
detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()
flann_params = dict(algorithm = 1,trees = 5)
flann = cv2.FlannBasedMatcher(flann_params,{})
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)
extract_bow.setVocabulary(voc)
dsc = extract_bow.compute(blurredImage,detect.detect(gray2))
clf2 = joblib.load("svmtrained.pkl")
predicted = clf2.predict(dsc)
print labels[predicted[0]]
fin = np.zeros([420,256])
print fin.shape
gray2.resize(350,256)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(gray2,"Predicted: " + labels[predicted[0]],(20,290), font, 0.5,(255,0,0),1)
cv2.imshow("Labelled",gray2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
