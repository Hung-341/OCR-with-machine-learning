# Configuration file for the OCR project

# Paths
IMAGE_DIR = "path_to_your_image_data"
LABEL_FILE = "path_to_your_label_data"

# HOG Parameters
CELL_SIZE = 8
BLOCK_SIZE = 2
NBINS = 9

# SVM Parameters
LEARNING_RATE = 0.001
LAMBDA_PARAM = 0.01
N_ITERS = 1000

# Image Preprocessing Parameters
IMAGE_SIZE = (32, 32)
MIN_SIZE = 30
DISK_SIZE = 3
