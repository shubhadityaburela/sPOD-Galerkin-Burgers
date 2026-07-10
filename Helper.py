
import cv2
import h5py
import numpy as np
import scipy.io as sio


def window_cut_pressure_data(t_start, t_end, pressure_sensor_array, t):
    idx = np.where((t >= t_start) & (t <= t_end))

    return pressure_sensor_array[:13, idx[0]], pressure_sensor_array[13:, idx[0]], t[idx[0]]



def extract_matrix_from_jpg():
    # 1. Load your pre-cropped image
    image_path = 'cropped.jpg'
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 2. MASK THE LEGEND BOX
    # ---------------------------------------------------------
    # Replace these with the coordinates of the legend box
    # *inside your newly cropped image*
    leg_x_top_left = 0       # How far from the left edge
    leg_y_top_left = 0       # How far from the top edge
    leg_x_bottom_right = 2000  # Right edge of the legend
    leg_y_bottom_right = 300  # Bottom edge of the legend

    # Create a blank mask and draw a solid white rectangle over the legend area
    legend_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    cv2.rectangle(legend_mask, (leg_x_top_left, leg_y_top_left), (leg_x_bottom_right, leg_y_bottom_right), 255, -1)

    # 3. Create masks for the Red and Cyan markers (for the rest of the image)
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 100, 100])
    mask_red = cv2.inRange(img_rgb, lower_red, upper_red)

    lower_cyan = np.array([0, 150, 150])
    upper_cyan = np.array([100, 255, 255])
    mask_cyan = cv2.inRange(img_rgb, lower_cyan, upper_cyan)

    # Combine the marker masks and expand them slightly
    combined_mask = cv2.bitwise_or(mask_red, mask_cyan)
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Add the massive legend mask to our combined mask!
    final_mask = cv2.bitwise_or(combined_mask, legend_mask)

    # 4. Erase the legend/markers and fill the gaps (Inpainting)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    clean_img = cv2.inpaint(img_bgr, final_mask, 25, cv2.INPAINT_TELEA)

    # 5. Convert to Grayscale & Normalize (0.0 to 1.0)
    relative_matrix = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    relative_matrix = relative_matrix / 255.0

    # 6. Save and plot
    print(f"Matrix extracted! Final Shape: {relative_matrix.shape}")

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # plt.imshow(relative_matrix, cmap='magma', aspect='auto')
    # plt.title('Cleaned Matrix (No Legend or Markers)')
    # plt.axis('off')
    # plt.show()

    return relative_matrix


def extract_matrix_from_matlab_file():
    mat_path = 'myles_data.mat'

    raw_numbers = []

    # 1. Read lines, skipping comments and grabbing numbers
    with open(mat_path, 'r') as f:
        for line in f:
            cleaned = line.strip()
            # Skip empty lines or text headers
            if not cleaned or cleaned.startswith('#') or 'name:' in cleaned or 'type:' in cleaned:
                continue

            # Extract numbers from the line
            raw_numbers.extend([float(x) for x in cleaned.split()])

    # 2. The first 3 numbers in the data stream are the dimensions: [306, 919, 3]
    # We pop them out to structure our array dynamically
    rows = int(raw_numbers[0])  # 306
    cols = int(raw_numbers[1])  # 919
    channels = int(raw_numbers[2])  # 3

    # The remaining numbers are the actual matrix elements
    matrix_data = np.array(raw_numbers[3:])

    # 3. Reshape using Fortran 'F' order (Column-major), which is how Octave dumps it
    img_mat = matrix_data.reshape((rows, cols, channels), order='F')
    print(f"Successfully loaded Octave matrix with shape: {img_mat.shape}")

    # 4. Collapse the 3 channels down to a 2D matrix by averaging them
    relative_matrix = np.mean(img_mat, axis=-1)

    print(f"Final 2D matrix shape for pcolormesh: {relative_matrix.shape}")
    return relative_matrix