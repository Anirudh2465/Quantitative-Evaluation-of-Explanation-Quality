import cv2
import numpy as np
import os
import re
from glob import glob

def preprocess_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob(os.path.join(input_folder, '*.jpg'))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, binary = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
        inpainted = cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)
        filename = os.path.basename(image_path)
        match = re.search(r'(\d+)', filename)
        if match:
            number = match.group(1)
            new_filename = f"{number}.jpg"
        else:
            new_filename = filename
        output_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(output_path, inpainted)

        print(f"Processed and saved: {output_path}")

    print("All images processed and saved.")