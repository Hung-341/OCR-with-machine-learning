from data.preprocess import preprocess_data
from model.model import HOGFeatureExtractor, SVM
from model.train import train_model

def main():
    # Preprocess the data
    image_dir = "path_to_your_image_data"
    label_file = "path_to_your_label_data"
    images, labels = preprocess_data(image_dir, label_file)

    # Train the model
    train_model(images, labels)

if __name__ == "__main__":
    main()
