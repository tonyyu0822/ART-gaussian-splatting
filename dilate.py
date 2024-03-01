import os
import cv2

input_folder = 'Dataset/flower/masks_og'
output_folder = 'Dataset/flower/masks_dilate'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.bmp')):
        # load mask
        mask_path = os.path.join(input_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # adjust size
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        output_mask_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_mask_path, dilated_mask)
