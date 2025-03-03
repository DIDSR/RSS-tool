import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPfunctions import bd2Fdesc, Fdesc2bd, imbinarize, FD_change, contour2mask


# Load the binary mask image
mask = cv2.imread('/home/shuyue.guan/Documents/RSS-tool/truth_mask.png', cv2.IMREAD_GRAYSCALE)
mask = imbinarize(mask) # binarize area image

# plt.figure()
# plt.scatter(idx_arr[:, 1], idx_arr[:, 0], marker='.')
# plt.gca().invert_yaxis()
# plt.show()

# Save or display the result
# cv2.imwrite('contour_image.png', contour_image)

fd_code = bd2Fdesc(mask)
CFcode = FD_change(fd_code,2,10,1)




re_img = Fdesc2bd(CFcode, mask.shape)
syn_img = contour2mask(re_img)

print(syn_img)
print(syn_img.shape)

plt.figure()
plt.imshow(syn_img, cmap='Greys_r', interpolation='none')
plt.show()
