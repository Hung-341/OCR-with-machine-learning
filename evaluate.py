import numpy as np
from data.preprocess import preprocess_data
from model.model import HOGFeatureExtractor, SVM

def evaluate_model(model, images, labels):
    # Extract HOG features
    hog = HOGFeatureExtractor()
    features = np.array([hog.extract(image) for image in images])

    # Make predictions
    predictions = model.predict(features)

    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

def main():
    # Load and preprocess the evaluation dataset
    image_dir = "path_to_your_test_image_data"
    label_file = "path_to_your_test_label_data"
    images, labels = preprocess_data(image_dir, label_file)

    # Load or train your model (Here it's assumed the model is trained already)
    model = SVM()
    model.fit(images, labels)  # Replace with actual model loading

    # Evaluate the model
    evaluate_model(model, images, labels)

if __name__ == "__main__":
    main()
