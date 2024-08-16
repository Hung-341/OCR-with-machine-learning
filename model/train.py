import numpy as np
from model.model import HOGFeatureExtractor, SVM

def load_data():
    # This function should be replaced with actual code to load your dataset
    # Here we create some dummy data for demonstration purposes
    # X: Images, y: Labels (e.g., 0 for 'A', 1 for 'B')
    num_samples = 100
    image_size = (32, 32)
    X = np.random.rand(num_samples, *image_size) * 255
    y = np.random.randint(0, 2, num_samples)
    return X, y

def train_model():
    # Load data
    X, y = load_data()

    # Feature extraction using HOG
    hog = HOGFeatureExtractor(cell_size=8, block_size=2, nbins=9)
    X_hog = np.array([hog.extract(image) for image in X])

    # Initialize SVM model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)

    # Training with progress display
    n_samples = X_hog.shape[0]
    print("Training SVM model...")
    for i in range(svm.n_iters):
        svm.fit(X_hog, y)

        # Display progress
        progress = (i + 1) / svm.n_iters
        bar_length = 40
        block = int(round(bar_length * progress))
        progress_bar = "#" * block + "-" * (bar_length - block)
        print(f"\rProgress: [{progress_bar}] {int(progress * 100)}%", end="")

    print("\nTraining complete.")

if __name__ == "__main__":
    train_model()
