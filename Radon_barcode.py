import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon

'''
img : input image
nrows,ncols: new size for normalization (recommended: nrows=ncols)
numRays: number of projections
'''
# Constants
nrows = 128
ncols = 128
numRays = 180

img = cv2.imread('/home/dell/Pictures/clock.jpg')
print img.shape
normImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
normImg = cv2.resize(normImg, (nrows, ncols))
cv2.imshow('filtered', normImg)
cv2.waitKey(0)

rayVector = np.arange(0., 180., (180.0/float(numRays)))
# print rayVector
RadonTransform = radon(normImg, theta=rayVector)
cv2.imshow('filtered', RadonTransform)
cv2.waitKey(0)

print RadonTransform.shape
print type(RadonTransform)
print RadonTransform

RadonTransformResized = cv2.resize(RadonTransform, (32,32))
cv2.imshow('filtered', RadonTransformResized)
cv2.waitKey(0)
np.savetxt("radonTransformresized.txt", RadonTransformResized)
norm_radon = cv2.normalize(RadonTransformResized, RadonTransformResized,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    #ksize = 31

    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sinusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold
    for ksize in np.arange(11,33,2):
	    for theta in np.arange(0, np.pi, np.pi / 8):
	        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
	                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
	        kern = cv2.getGaborKernel(**params)
	        kern /= 1.5*kern.sum()
	        #print kern
	        filters.append((kern,params))
    return filters

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    barcode = []
    i=0
    for kern,params in filters:
        print i
        i+=1
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        #cv2.imshow('filtered', fimg)
        #cv2.waitKey(0)
        fimg = np.absolute(fimg)
        #downsample
        fimgVector = fimg.flatten()
        threshold = np.median(fimgVector)
        print threshold
        fimgVector = fimgVector > threshold
        #print fimgVector
        barcode = np.append(barcode, fimgVector)
    barcode = np.asarray(barcode, dtype=np.uint8)
    print barcode.shape
    return barcode

filters = build_filters()
barcode = process(norm_radon, filters)
print barcode

# plot barcode
fig = plt.figure()
x = np.reshape(barcode, (1, np.product(barcode.shape))) # horizontal barcode
#x = np.reshape(barcode, (np.product(barcode.shape), 1)) # vertical barcode
axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')
ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax.imshow(x, **barprops)
ax.set_title('Radon Barcode')
plt.show()