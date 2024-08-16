import numpy as np
import os
from skimage import io, color, transform, filters, morphology, exposure

def preprocess_data(image_dir, label_file):
    images = []
    labels = []

    # Load and preprocess images
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            image = io.imread(image_path)

            # Step 1: Convert to grayscale
            image = color.rgb2gray(image)

            # Step 2: Resize to a fixed size (e.g., 32x32)
            image = transform.resize(image, (32, 32))

            # Step 3: Binarize the image (thresholding)
            thresh = filters.threshold_otsu(image)
            binary_image = image > thresh

            # Step 4: Morphological operations to clean up noise
            # Remove small objects and close gaps
            cleaned_image = morphology.remove_small_objects(binary_image, min_size=30)
            cleaned_image = morphology.binary_closing(cleaned_image, morphology.disk(3))

            # Step 5: Normalize pixel values to range [0, 1]
            normalized_image = exposure.rescale_intensity(cleaned_image, out_range=(0, 1))

            images.append(normalized_image)

    images = np.array(images)

    # Load labels (assumed to be in a text file with one label per line)
    with open(label_file, 'r') as file:
        labels = [int(line.strip()) for line in file.readlines()]

    labels = np.array(labels)

    return images, labels
