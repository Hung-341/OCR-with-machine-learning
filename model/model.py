import numpy as np

class HOGFeatureExtractor:
    def __init__(self, cell_size=8, block_size=2, nbins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins

    def compute_gradients(self, image):
        # Calculate gradients using simple finite difference
        gx = np.zeros_like(image, dtype=np.float32)
        gy = np.zeros_like(image, dtype=np.float32)
        gx[:, :-1] = np.diff(image, n=1, axis=1)
        gy[:-1, :] = np.diff(image, n=1, axis=0)
        return gx, gy

    def compute_cell_histogram(self, gx, gy):
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx) * (180.0 / np.pi) % 180
        bins = np.int32(self.nbins * angle / 180)

        # Divide magnitude into cells and compute histogram for each cell
        cell_histogram = np.zeros((gx.shape[0] // self.cell_size, gx.shape[1] // self.cell_size, self.nbins))
        for i in range(cell_histogram.shape[0]):
            for j in range(cell_histogram.shape[1]):
                cell_magnitude = magnitude[i*self.cell_size:(i+1)*self.cell_size,
                                           j*self.cell_size:(j+1)*self.cell_size]
                cell_bins = bins[i*self.cell_size:(i+1)*self.cell_size,
                                 j*self.cell_size:(j+1)*self.cell_size]
                cell_histogram[i, j] = np.bincount(cell_bins.ravel(), weights=cell_magnitude.ravel(), minlength=self.nbins)

        return cell_histogram

    def block_normalize(self, cell_histogram):
        # Normalize the histograms within the blocks
        blocks = []
        for i in range(0, cell_histogram.shape[0] - self.block_size + 1):
            for j in range(0, cell_histogram.shape[1] - self.block_size + 1):
                block = cell_histogram[i:i+self.block_size, j:j+self.block_size].ravel()
                block /= np.linalg.norm(block) + 1e-6
                blocks.append(block)
        return np.concatenate(blocks)

    def extract(self, image):
        gx, gy = self.compute_gradients(image)
        cell_histogram = self.compute_cell_histogram(gx, gy)
        hog_features = self.block_normalize(cell_histogram)
        return hog_features


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
