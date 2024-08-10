import numpy as np
import cv2 as cv

def reduce_2d(data1, data2, axis):
    c = 0
    while True:
        if axis == 2:
            slice_data = data1[:, :, 0 if c == 0 else -1].flatten()
        elif axis == 1:
            slice_data = data1[:, 0 if c == 0 else -1, :].flatten()
        else:
            slice_data = data1[0 if c == 0 else -1, :, :].flatten()

        if np.max(slice_data) == np.min(slice_data):  # Black slice
            if c == 0:
                data1 = np.delete(data1, 0, axis=axis)
                data2 = np.delete(data2, 0, axis=axis)
            else:
                data1 = np.delete(data1, -1, axis=axis)
                data2 = np.delete(data2, -1, axis=axis)
        else:
            if c == 0:
                c = 1
            else:
                break
    return data1, data2

def flip(image, mask, flip_code):
    if flip_code == 2:
        return image, mask
    return cv.flip(image, flip_code), cv.flip(mask, flip_code)

def blur(image, blur_type):
    if blur_type == 0:
        return image

    blur_option = np.random.randint(3)
    if blur_option == 0:
        return cv.GaussianBlur(image, (11, 11), 0)
    elif blur_option == 1:
        return cv.GaussianBlur(image, (15, 1), 0)
    else:
        return cv.GaussianBlur(image, (1, 15), 0)