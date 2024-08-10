import numpy as np
import os
import nibabel as nib
from ipywidgets import interact, interactive, fixed, AppLayout
import matplotlib.pyplot as plt

def visualize_2d(img):
    """Visualize a 2D slice of an image."""
    max_val = np.max(img)
    if max_val > 0 and not (max_val <= 255 and isinstance(max_val, (np.uint8, int, np.int8, np.int16))):
        img = np.array(img * 255 / max_val, dtype=np.uint8)
    else:
        img = np.array(img, dtype=np.uint8)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def explore_image(layer, n, data):
    """Explore and visualize a 3D MRI along each axis."""
    if n == 0:
        visualize_2d(data[layer, :, :])  # Sagittal plane
    elif n == 1:
        visualize_2d(data[:, layer, :])  # Coronal plane
    elif n == 2:
        visualize_2d(data[:, :, layer])  # Axial plane
    return layer

def return_3d(select_file, c=0, x=0):
    """Create an interactive 3D visualization with sliders for each axis."""
    if c == 1:
        data1 = nib.load(os.path.join(x, select_file)).get_fdata() if x else nib.load(select_file).get_fdata()
    else:
        data1 = select_file
    
    max_val = np.max(data1)
    data = np.array((data1 * 255 / max_val), dtype=np.uint8)

    i1 = interactive(explore_image, layer=(0, data.shape[0] - 1), n=fixed(0), data=fixed(data))
    i2 = interactive(explore_image, layer=(0, data.shape[1] - 1), n=fixed(1), data=fixed(data))
    i3 = interactive(explore_image, layer=(0, data.shape[2] - 1), n=fixed(2), data=fixed(data))

    layout = AppLayout(header=None, left_sidebar=i1, center=i2, right_sidebar=i3, footer=None, pane_widths=[1, 1, 1])
    display(layout)

def visualize_3d():
    """Create an interface for visualizing 3D MRI along three axes."""
    path = input('Enter path containing image folder: ').strip()
    if not path.endswith('/'):
        path += '/'
    file_list = os.listdir(path)
    interact(return_3d, select_file=file_list, c=fixed(1), x=fixed(path))

def brain_part_focus(data1, data2):
    """Focus on and visualize specific brain parts from segmented MRI data."""
    brain_parts = [
        "Intracranial space and extra-axial CSF spaces",
        "Gray matter",
        "White matter",
        "Ventricles",
        "Cerebellum",
        "Deep gray matter",
        "Brainstem and spinal cord",
        "Segmented brain without noise"
    ]
    print('Enter the segmented brain part to view:')
    for idx, part in enumerate(brain_parts, 1):
        print(f"{idx}. {part}")

    while True:
        choice = int(input('\nEnter your choice: '))
        if 1 <= choice <= 8:
            break
        print('Invalid choice. Retry!')

    if choice == 8:
        focused_data = np.where(data2 > 0, data1, 0)
    else:
        focused_data = np.where(data2 == choice, data1, 0)

    visualize_2d(focused_data)
