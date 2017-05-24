from skimage import color, io
from skimage.feature import match_template
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

import skimage.transform as tsf
import skimage.segmentation as seg
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 'small'

# The x and y number of subplots, and the index of the subplot
m, n, i = 2, 3, 1

im1 = io.imread('54.jpg')
ind1 = np.array(im1)
print("IM1 shape: ", ind1.shape)
plt.subplot(m, n, i)
plt.imshow(im1)
plt.title('Original Image in RGB color')

i += 1
im2 = color.rgb2gray(im1)
ind2 = np.array(im2)
print("IM2 shape: ", ind2.shape)
plt.subplot(m, n, i)
plt.imshow(im2)
plt.title('Converted Image in Gray')

i += 1
im3 = tsf.resize(im2, (240, 320), order=1)
ind3 = np.array(im3)
print("IM3 shape: ", ind3.shape)
plt.subplot(m, n, i)
plt.imshow(im3)
plt.title('Resized (240x320) Image in Gray')

print(ind2[160:200, 120:160])

i += 1
im4 = tsf.resize(im2, (120, 160), order=1)
plt.subplot(m, n, i)
plt.imshow(im4)
plt.title('Resized (120x160) Image in Gray')

i += 1
im6 = seg.find_boundaries(im2, connectivity=1, mode='inner', background=0)
plt.subplot(m, n, i)
plt.imshow(im6)
plt.title('Segmented Boolean Value Image')

#i += 1
#distance = ndi.distance_transform_edt(im2)
#local_maxi = peak_local_max(distance, indices=False, labels=im2)
#markers = ndi.label(local_maxi)[0]
#labels = watershed(-distance, markers, mask=im2)

# result = match_template(im1, template=, mode)
#plt.subplot(m, n, i)
#plt.imshow(labels)
#plt.title('Watershed Segmented Image')

plt.show()