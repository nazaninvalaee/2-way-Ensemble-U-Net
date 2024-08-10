import numpy as np
import os
import nibabel as nib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Dataset.preprocessing import reduce_2d, flip, blur

def split_dataset(input_mri, output_mri, test_size):
    input_mri = np.array(input_mri, dtype=np.uint8)
    output_mri = np.array(output_mri, dtype=np.uint8)

    if test_size == 0:
        input_mri = np.array(input_mri / 128, dtype=np.float16)
        return input_mri, output_mri

    X_train, X_test, y_train, y_test = train_test_split(input_mri, output_mri, test_size=test_size, random_state=38)
    X_train = np.array(X_train / 128, dtype=np.float16)
    X_test = np.array(X_test / 128, dtype=np.float16)
    return X_train, X_test, y_train, y_test

def create_dataset(input_path, output_path, slices_per_axis=40, test_size=0.05):
    if not input_path.endswith('/'):
        input_path += '/'
    if not output_path.endswith('/'):
        output_path += '/'

    input_files = os.listdir(input_path)
    num_files = len(input_files)

    non_blur_files = {
        'sub-009', 'sub-005', 'sub-008', 'sub-007', 'sub-004', 'sub-002', 'sub-015', 'sub-023', 'sub-016',
        'sub-017', 'sub-022', 'sub-021', 'sub-020', 'sub-062', 'sub-071', 'sub-078'
    }

    input_mri = []
    output_mri = []

    for file_name in tqdm(input_files, desc="Processing Files", ncols=75):
        input_file_path = os.path.join(input_path, file_name)
        output_file_name = file_name.replace('_T2w.nii.gz', '_dseg.nii.gz')
        output_file_path = os.path.join(output_path, output_file_name)

        input_data = nib.load(input_file_path).get_fdata()
        output_data = nib.load(output_file_path).get_fdata().astype(np.uint8)

        axes = [(0, input_data.shape[0]), (1, input_data.shape[1]), (2, input_data.shape[2])]
        for axis, size in axes:
            input_slices, output_slices = reduce_2d(input_data, output_data, axis)
            slice_indices = list(range(0, size, max(size // slices_per_axis, 1)))

            for idx in slice_indices:
                if axis == 0:
                    slice_input = input_slices[idx, :, :]
                    slice_output = output_slices[idx, :, :]
                elif axis == 1:
                    slice_input = input_slices[:, idx, :]
                    slice_output = output_slices[:, idx, :]
                else:
                    slice_input = input_slices[:, :, idx]
                    slice_output = output_slices[:, :, idx]

                flip_type = np.random.randint(-1, 3)
                slice_input, slice_output = flip(slice_input, slice_output, flip_type)

                if file_name[:7] not in non_blur_files:
                    blur_type = np.random.randint(2)
                    slice_input = blur(slice_input, blur_type)

                max_intensity = slice_input.max()
                if max_intensity > 0:
                    slice_input = (slice_input * 255 / max_intensity).astype(np.uint8)

                input_mri.append(slice_input)
                output_mri.append(slice_output)

    return split_dataset(input_mri, output_mri, test_size)
