import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

def augment_images(image_folder, output_folder, label_file, output_label_file):
    IMAGE_FOLDER = image_folder
    OUTPUT_FOLDER = output_folder
    LABEL_FILE = label_file
    OUTPUT_LABEL_FILE = output_label_file

    ENHANCEMENT_TECHNIQUES = [
        'original',
        'histogram_eq',
        'clahe',
        'gaussian_blur',
        'edge_enchancement'
    ]

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    df = pd.read_csv(LABEL_FILE)
    df.columns = [col.strip() for col in df.columns]

    max_id = df['Image ID'].max()
    next_id = max_id + 1

    augmented_data = []

    def apply_adaptive_threshold(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else thresh

    def apply_clahe(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else enhanced

    def apply_negative(image):
        return 255 - image

    def apply_gaussian_blur(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def apply_sharpening(image):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def apply_histogram_eq(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else equalized

    def apply_edge_enhancement(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 100, 200)
        if len(image.shape) == 3:
            color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(image, 1, color_edges, 0.5, 0)
        else:
            return cv2.addWeighted(gray, 0.7, edges, 0.3, 0)

    def enhance_image(image, technique):
        if technique == 'original':
            return image
        elif technique == 'adaptive_thresh':
            return apply_adaptive_threshold(image)
        elif technique == 'clahe':
            return apply_clahe(image)
        elif technique == 'negative':
            return apply_negative(image)
        elif technique == 'gaussian_blur':
            return apply_gaussian_blur(image)
        elif technique == 'sharpening':
            return apply_sharpening(image)
        elif technique == 'histogram_eq':
            return apply_histogram_eq(image)
        elif technique == 'edge_enhancement':
            return apply_edge_enhancement(image)
        else:
            return image

    print("Starting image enhancement augmentation...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['Image ID']
        img_path = os.path.join(IMAGE_FOLDER, f"{int(img_id)}.jpg")
        labels = {
            'Zoom': row['Zoom'],
            'Sagital': row['Sagital'],
            'Neutral': row['Neutral'],
            'Caliper': row['Caliper']
        }

        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        for technique in ENHANCEMENT_TECHNIQUES:
            if technique == 'original':
                original_filename = f"{int(img_id)}.jpg"
                new_path = os.path.join(OUTPUT_FOLDER, original_filename)
                cv2.imwrite(new_path, image)
                augmented_data.append([int(img_id), labels['Zoom'], labels['Sagital'], labels['Neutral'], labels['Caliper']])
            else:
                enhanced = enhance_image(image.copy(), technique)
                new_filename = f"{next_id}.jpg"
                new_path = os.path.join(OUTPUT_FOLDER, new_filename)
                cv2.imwrite(new_path, enhanced)
                augmented_data.append([int(next_id), labels['Zoom'], labels['Sagital'], labels['Neutral'], labels['Caliper']])
                next_id += 1

    augmented_df = pd.DataFrame(augmented_data, columns=['Image ID', 'Zoom', 'Sagital', 'Neutral', 'Caliper'])
    augmented_df.to_csv(OUTPUT_LABEL_FILE, index=False)

    print(f"Augmentation completed. Total images: {len(augmented_data)}")
    print(f"Original images: {len(df)}, New images: {len(augmented_data) - len(df)}")
    print(f"Augmented images saved to {OUTPUT_FOLDER}")
    print(f"Updated labels saved to {OUTPUT_LABEL_FILE}")
