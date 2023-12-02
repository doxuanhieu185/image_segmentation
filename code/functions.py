import cv2 as cv2
import numpy as np

def image_resize_dimensions(image, ratio=1.0):
    scale_percent = ratio
    image_width = int(image.shape[1] * scale_percent)
    image_height = int(image.shape[0] * scale_percent)
    dim = (image_width, image_height)

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def image_threshold(image, threshold=127, adaptive=False, otsu=False, kernel_size=3):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if adaptive:
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, kernel_size, 0)
    elif otsu:
        ret_val, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return image
    else:
        ret_val, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        return image


def image_sobel_gradient(image):
    if len(image.shape) == 3:
        full_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        full_grey = np.copy(image)

    kernel_size = 3
    gradient_x = cv2.Sobel(full_grey, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=kernel_size)
    gradient_y = cv2.Sobel(full_grey, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=kernel_size)

    gradient_x = cv2.convertScaleAbs(gradient_x)  # Converting back to an 8-bit integer so OpenCV can operate
    gradient_y = cv2.convertScaleAbs(gradient_y)

    combined = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)  # combines into a single image

    return combined


def image_canny_edges(image):
    med_val = np.median(image)
    lower = int(max(0, .7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(image=image, threshold1=lower, threshold2=upper)
    return edges


def image_kmean_segmentation(image, k):
    color = np.array(
        [[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [119, 159, 68],
         [97, 220, 151], [228, 164, 166], [191, 75, 184], [229, 40, 22], [243, 177, 234], [96, 43, 149], [113, 39, 186],
         [131, 227, 133], [203, 15, 240], [181, 110, 167], [187, 63, 206], [202, 70, 10], [43, 146, 61], [185, 10, 209],
         [28, 79, 72], [75, 183, 187], [135, 125, 93], [253, 76, 44], [212, 9, 132], [126, 215, 56], [84, 198, 179],
         [115, 104, 183], [243, 188, 33], [29, 150, 16], [6, 224, 62], [150, 92, 249], [249, 106, 81], [15, 91, 39],
         [51, 210, 91], [110, 81, 133], [102, 155, 71], [135, 35, 102], [165, 157, 110], [121, 221, 60], [152, 193, 20],
         [163, 222, 237], [177, 97, 149], [55, 23, 226], [114, 54, 212], [68, 73, 88], [128, 53, 147], [214, 19, 144],
         [98, 165, 163], [53, 170, 70], [108, 15, 97], [5, 250, 78], [65, 6, 215], [152, 55, 172], [101, 198, 200],
         [87, 109, 216], [233, 240, 202], [46, 44, 128], [184, 247, 112], [75, 33, 136], [189, 143, 210],
         [123, 90, 167], [83, 35, 232], [182, 187, 68], [92, 199, 225], [182, 56, 22], [122, 223, 138], [233, 166, 43],
         [113, 81, 231], [245, 189, 2], [11, 127, 78], [118, 82, 157], [41, 47, 48], [113, 224, 107], [156, 7, 203],
         [25, 228, 33], [104, 141, 56], [74, 7, 244], [28, 85, 27], [45, 109, 211], [228, 255, 8], [23, 194, 114],
         [32, 225, 32], [25, 30, 126], [83, 163, 112], [137, 143, 65], [20, 52, 218], [167, 13, 230], [38, 0, 117],
         [70, 102, 249], [93, 20, 233], [31, 248, 67]])

    color = np.uint8(color)
    image_2d = image.reshape((-1, 3))
    image_2d = np.float32(image_2d)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(image_2d, k, None, criteria=criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS)
    segmented_data = color[label.flatten()]

    return segmented_data.reshape(image.shape)