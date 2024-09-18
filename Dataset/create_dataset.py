import numpy as np
import os
import nibabel as nib
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur
from skimage.transform import resize

# Function for splitting the dataset into train and test set and normalizing the values
def split_dataset(input_mri, output_mri, s):
    input_mri = np.array(input_mri, dtype=np.uint8)
    output_mri = np.array(output_mri, dtype=np.uint8)
    
    if s == 0:
        # Normalizing the input to the range 0 to 2
        input_mri = np.array(input_mri / 128, dtype=np.float16)
        return input_mri, output_mri
    else:  
        # Splitting up the dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(input_mri, output_mri, test_size=s, random_state=38)
        
        del input_mri, output_mri
        
        # Normalizing the input to the range 0 to 2
        X_train = np.array(X_train / 128, dtype=np.float16)
        X_test = np.array(X_test / 128, dtype=np.float16)
        
        return X_train, X_test, y_train, y_test

'''  
     Function to create the dataset for both training and testing the model
     It will now include multi-scale images, so for each image slice, additional downscaled versions are added.
     It will also prepare the data for Edge-Aware Loss by highlighting boundaries.
'''
def create_dataset(path1, path2, n=40, s=0.05):

    if not path1.endswith('/'):
        path1 += '/'
    if not path2.endswith('/'):
        path2 += '/'

    l = os.listdir(path1)
    num = len(l)

    non_blur = ['sub-009', 'sub-005', 'sub-008', 'sub-007', 'sub-004', 'sub-002', 'sub-015', 'sub-023', 'sub-016', 'sub-017', 
                'sub-022', 'sub-021', 'sub-020', 'sub-062', 'sub-071', 'sub-078']

    input_mri = []
    output_mri = []

    for i in tqdm(range(num), desc="Executing", ncols=75):
        f1 = l[i]                      
        f2 = f1[:-10]+'dseg'+f1[-7:]    

        # Load input and output MRI data
        data1 = nib.load(path1 + f1).get_fdata()
        data2 = nib.load(path2 + f2).get_fdata()
        data2 = np.array(data2, dtype=np.uint8)

        # Process data across all axes
        for axis in range(3):
            data_1_slice, data_2_slice = reduce_2d(data1, data2, axis)
            slice_shape = np.asarray(data_1_slice).shape[axis]
            slice_step = int(slice_shape / n)
            selected_slices = list(range(0, slice_shape, slice_step))

            # Extract multi-scale slices and apply transformations
            for j in selected_slices:
                if axis == 0:
                    d1, d2 = data_1_slice[j, :, :], data_2_slice[j, :, :]
                elif axis == 1:
                    d1, d2 = data_1_slice[:, j, :], data_2_slice[:, j, :]
                else:
                    d1, d2 = data_1_slice[:, :, j], data_2_slice[:, :, j]

                # Apply transformations (flip, blur)
                f = np.random.randint(-1, 3)
                d1, d2 = flip(d1, d2, f)

                if f1[:7] not in non_blur:
                    f = np.random.randint(2)
                    d1 = blur(d1, f)

                # Add multi-scale versions
                multi_scales = [d1, resize(d1, (d1.shape[0] // 2, d1.shape[1] // 2)), resize(d1, (d1.shape[0] // 4, d1.shape[1] // 4))]
                input_mri.extend(multi_scales)
                output_mri.extend([d2] * len(multi_scales))  # Repeat segmentation for each scale

                # Clip and normalize input
                d1 = np.array(d1 * 255 / np.max(d1), dtype=np.uint8)
                input_mri.append(d1)
                output_mri.append(d2)

    return split_dataset(input_mri, output_mri, s)
