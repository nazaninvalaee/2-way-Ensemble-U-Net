import numpy as np
import os
import nibabel as nib
from ipywidgets import interact, interactive, fixed, AppLayout
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

# Function to visualize a 2D slice and apply optional colormap
def visualize_2d(img, cmap='gray'):
    m = max(np.reshape(img, (-1)))
    l = [np.uint8, int, np.int8, np.int16]

    if m > 0 and not (m <= 255 and (type(m) in l)):
        img = np.array(img * 255 / m, dtype=np.uint8)
    else:
        img = np.array(img, dtype=np.uint8)

    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

# Function to visualize the 3D MRI along each axis
def explore_image(layer, n, data, cmap='gray'):
    if n == 0:
        visualize_2d(data[layer, :, :], cmap)  # Saggital plane
    elif n == 1:
        visualize_2d(data[:, layer, :], cmap)  # Coronal plane
    elif n == 2:
        visualize_2d(data[:, :, layer], cmap)  # Horizontal plane
    return layer

# Creating a slider to select the layer to visualize in each axis
def return_3d(select_file, c=0, x=0, cmap='gray'):
    if c == 1:  # Load 3D MRI from file
        data1 = nib.load(x + select_file).get_fdata() if x else nib.load(select_file).get_fdata()
    else:  # File already loaded
        data1 = select_file

    m = max(data1.reshape(-1))
    data = np.array((data1 * 255 / m), dtype=np.uint8)  # Normalize to 0-255

    # Creating 3 interactive sliders for each axis
    i1 = interactive(explore_image, layer=(0, data.shape[0] - 1), n=fixed(0), data=fixed(data), cmap=fixed(cmap))
    i2 = interactive(explore_image, layer=(0, data.shape[1] - 1), n=fixed(1), data=fixed(data), cmap=fixed(cmap))
    i3 = interactive(explore_image, layer=(0, data.shape[2] - 1), n=fixed(2), data=fixed(data), cmap=fixed(cmap))

    # Layout to visualize all axes side by side
    layout = AppLayout(header=None, left_sidebar=i1, center=i2, right_sidebar=i3, footer=None, pane_widths=[1, 1, 1])
    display(layout)

# Function to create an interface for visualizing 3D MRI with optional colormap
def visualize_3d():
    x = input('Enter path containing image folder: ')  # Take folder path input
    if not x.endswith('/'):
        x = x + '/'
    l = os.listdir(x)  # List all files in the folder
    cmap = input('Enter colormap (default "gray"): ') or 'gray'
    interact(return_3d, select_file=l, c=fixed(1), x=fixed(x), cmap=fixed(cmap))  # Dropdown to select 3D image

# Overlay predictions on original MRI
def overlay_predictions(mri_img, pred_img, alpha=0.4, cmap='jet'):
    plt.imshow(mri_img, cmap='gray')
    plt.imshow(pred_img, cmap=cmap, alpha=alpha)  # Overlay prediction with transparency
    plt.axis('off')
    plt.show()

# Visualize and focus on a particular brain part (segmentation)
def brain_part_focus(data1, data2):
    print('Enter the segmented brain part to view:\n')
    print('1. Intracranial space and extra-axial CSF spaces')
    print('2. Gray matter')
    print('3. White matter')
    print('4. Ventricles')
    print('5. Cerebellum')
    print('6. Deep gray matter')
    print('7. Brainstem and spinal cord')
    print('8. Segmented brain without noise')

    while True:
        i = int(input('\nEnter your choice: '))
        if i < 1 or i > 8:
            print('Invalid choice. Retry!')
        else:
            break

    if i == 8:
        d1 = np.where(data2 > 0, data1, 0)  # Keep all brain parts
    else:
        d1 = np.where(data2 == i, data1, 0)  # Keep only the selected brain part

    visualize_2d(d1)

# Visualize boundaries in segmentation
def visualize_boundaries(segmentation):
    boundaries = find_boundaries(segmentation, mode='outer')
    plt.imshow(boundaries, cmap='hot')
    plt.axis('off')
    plt.show()

# Save a visualization to a file
def save_visualization(img, filename, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved visualization as {filename}')
