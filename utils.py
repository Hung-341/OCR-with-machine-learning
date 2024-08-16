import pickle

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def visualize_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.show()
