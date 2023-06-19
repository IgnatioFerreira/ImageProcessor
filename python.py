# import cv2
import numpy as np
# from sklearn.cluster import KMeans

# img = cv2.imread('input.jpg')


# def reduce_grayscale():
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     max_val = gray_img.max()
#     min_val = gray_img.min()
#     levels = 3
#     range_val = max_val - min_val
#     if range_val == 0:
#         return gray_img
#     level_size = range_val / levels
#     gray_img = (((gray_img - min_val) / range_val) * levels).astype(np.uint8) * level_size + min_val
#     cv2.imwrite('output_grayscale_adaptive.png', gray_img)

# def quantize_color():
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hue = hsv[:, :, 0].flatten().reshape(-1, 1)
#     kmeans = KMeans(n_clusters=3)
#     labels = kmeans.fit_predict(hue)
#     quantized_hue = kmeans.cluster_centers_[labels].reshape(hsv[:, :, 0].shape)
#     hsv[:, :, 0] = quantized_hue
#     img_quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     cv2.imwrite('output_quantized_color.png', img_quantized)

# shift_hue()
# reduce_grayscale()
# quantize_color()

import cv2
from skimage.segmentation import slic
from skimage.color import label2rgb




# Load the image
image = cv2.imread('input.jpg')
img = cv2.imread('input.jpg')

def segment_image():
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the SLIC algorithm to the image to generate 2000 superpixels, with a compactness of 10 and a sigma of 3
    segments = slic(image, n_segments=2000, compactness=25, sigma=1)

    # Convert the segments to a RGB image using median color averaging
    segmented_image = label2rgb(segments, image, kind='avg')

    # Save the segmented image
    cv2.imwrite('output.png', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

def shift_hue():
    hue_shift = 60
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
    hsv = hsv.astype(np.uint8)
    img_shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('hue_shifted2.png', img_shifted)

def quantize_brightness():
    levels = 5  # Number of quantization levels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = ((hsv[:,:,2]/255)*levels).astype(np.uint8)*(255/(levels-1))
    img_bright_quant = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('output_brightness_quantized.png', img_bright_quant)

# shift_hue()
quantize_brightness()