import numpy as np
import cv2 as cv

# Function to remove all the black slices (i.e. removing the 2D slices completely filled with 0)
def reduce_2d(data1, data2, n):
    # Same as your original reduce_2d function
    # (omitted here for brevity)
    return data1, data2


# Flipping the images to introduce some translational independence
def flip(d1, d2, i):  # Possible values of i are (-1, 0, 1, 2)
    if i == 2:
        return d1, d2  # No flipping
    else:
        return cv.flip(d1, i), cv.flip(d2, i)  # value of i determines the flip


# Blurring the images to account for blurring or stretching due to fetal movement
def blur(x, i):  # Possible values of i are (0, 1, 2)
    if i == 0:  # No blurring
        return x
    else:  # Apply some form of blurring based on a random choice
        f = np.random.randint(3)
        if f == 0:
            return cv.GaussianBlur(x, (11, 11), 0)  # Normal Blurring
        elif f == 1:
            return cv.GaussianBlur(x, (15, 1), 0)  # Horizontal stretching
        else:
            return cv.GaussianBlur(x, (1, 15), 0)  # Vertical stretching


# New: Mask parts of the image for self-supervised learning (e.g., inpainting)
def mask_image(image, mask_size=(50, 50)):
    """
    Applies a mask over random regions of the image. The masked regions will have zero values.
    This can be used for self-supervised tasks such as inpainting.
    """
    h, w = image.shape
    mask_x = np.random.randint(0, w - mask_size[0])
    mask_y = np.random.randint(0, h - mask_size[1])
    
    masked_image = image.copy()
    masked_image[mask_y:mask_y + mask_size[1], mask_x:mask_x + mask_size[0]] = 0  # Apply mask
    
    return masked_image


# New: Extract random patches for self-supervised patch prediction
def extract_patch(image, patch_size=(50, 50)):
    """
    Extracts a random patch from the image. This can be used for self-supervised tasks such as patch prediction.
    """
    h, w = image.shape
    patch_x = np.random.randint(0, w - patch_size[0])
    patch_y = np.random.randint(0, h - patch_size[1])
    
    patch = image[patch_y:patch_y + patch_size[1], patch_x:patch_x + patch_size[0]]
    return patch
