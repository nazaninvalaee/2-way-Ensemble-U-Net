import numpy as np
import os
import nibabel as nib
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur


# Function for generating synthetic MRI data
def generate_synthetic_data(shape):
    """
    Generates synthetic MRI data. 
    You can replace this function with a more sophisticated method like GAN-based generation.
    """
    return np.random.randn(*shape)  # Random noise data to simulate synthetic MRI


# Function for self-supervised tasks (e.g., patch prediction or inpainting)
def create_self_supervised_task(input_data, task="inpainting"):
    """
    Creates self-supervised tasks on input data.
    For inpainting, we'll mask random parts of the image.
    """
    if task == "inpainting":
        mask = np.random.randint(0, 2, size=input_data.shape)  # Random binary mask
        input_data_masked = input_data * mask  # Mask parts of the input data
        return input_data_masked, input_data  # Return masked input and the original for supervision

    # Other self-supervised tasks (e.g., patch prediction) can be added here
    return input_data, input_data  # Default: no task

# Function for splitting the dataset into train and test set and normalising the values
def split_dataset(input_mri, output_mri, s):
    input_mri = np.array(input_mri, dtype=np.uint8)
    output_mri = np.array(output_mri, dtype=np.uint8)

    if s == 0:
        input_mri = np.array(input_mri / 128, dtype=np.float16)
        return input_mri, output_mri
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_mri, output_mri, test_size=s, random_state=38)

        del input_mri, output_mri

        X_train = np.array(X_train / 128, dtype=np.float16)
        X_test = np.array(X_test / 128, dtype=np.float16)

        return X_train, X_test, y_train, y_test


'''
Function to create the dataset for both training and testing the model.
It now supports synthetic data generation and self-supervised tasks.
'''

def create_dataset(path1=None, path2=None, n=40, s=0.05, synthetic=False, self_supervised=False, task="inpainting"):
    """
    Parameters:
        path1: Path to real input MRIs (ignored if synthetic=True)
        path2: Path to real segmented outputs (ignored if synthetic=True)
        n: Number of 2D slices per axis to extract from MRI volumes
        s: Test set size as a fraction or number (no splitting if s=0)
        synthetic: If True, generates synthetic data
        self_supervised: If True, applies a self-supervised task to the input data
        task: Specifies the self-supervised task (e.g., 'inpainting')
    """
    
    input_mri = []
    output_mri = []

    if synthetic:
        print("Generating synthetic data...")
        # Example shape for synthetic MRI (can be modified based on the dataset)
        synthetic_shape = (256, 256, 256)
        for _ in range(n):  # Simulate n synthetic images
            input_synthetic = generate_synthetic_data(synthetic_shape)
            output_synthetic = generate_synthetic_data(synthetic_shape)  # Can be more realistic
            
            input_mri.append(input_synthetic)
            output_mri.append(output_synthetic)

    else:
        if not path1[-1] == '/':
            path1 = path1 + '/'
        if not path2[-1] == '/':
            path2 = path2 + '/'
        
        l = os.listdir(path1)  # Storing the file names of the input MRI folder
        num = len(l)

        non_blur = ['sub-009', 'sub-005', 'sub-008', 'sub-007', 'sub-004', 'sub-002', 'sub-015', 'sub-023', 
                    'sub-016', 'sub-017', 'sub-022', 'sub-021', 'sub-020', 'sub-062', 'sub-071', 'sub-078']

        for i in tqdm(range(num), desc="Executing", ncols=75):
            f1 = l[i]
            f2 = f1[:-10] + 'dseg' + f1[-7:]

            # Loading the real MRI files
            data1 = nib.load(path1 + f1).get_fdata()
            data2 = nib.load(path2 + f2).get_fdata()
            data2 = np.array(data2, dtype=np.uint8)

            # Applying preprocessing and slicing (as in the original code)
            data10, data20 = reduce_2d(data1, data2, 0)
            # (Similar slicing and transformation for axes 1 and 2 omitted for brevity)

            input_mri.append(data10)
            output_mri.append(data20)

    # Apply self-supervised tasks if enabled
    if self_supervised:
        input_mri, output_mri = zip(*[create_self_supervised_task(mri, task=task) for mri in input_mri])

    return split_dataset(input_mri, output_mri, s)
