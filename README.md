# OCR-with-machine-learning

## Overview

This project implements an Optical Character Recognition (OCR) system with machine learning, which extracts text from images and converts it into editable text. The project utilizes machine learning techniques to recognize and process various types of text from images.

## Features

- **Image Preprocessing**: Enhance image quality through noise reduction, thresholding, and other preprocessing techniques.
- **Text Detection**: Identify and isolate text regions within the image using advanced algorithms.
- **Text Recognition**: Convert the detected text regions into editable text using a trained machine learning model.
- **Output Formatting**: Provide the recognized text in a structured format, such as plain text or JSON.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hung-341/OCR-with-machine-learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd OCR-with-machine-learning
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your input images in the `input_images` directory.
2. Run the OCR process:
   ```bash
   python main.py --input_dir input_images --output_dir output_texts
   ```
3. The recognized text will be saved in the `output_texts` directory.

## Configuration

- You can customize the image preprocessing and text recognition settings by modifying the `config.yaml` file.
- The model used for text recognition can be swapped out by specifying a different model path in the configuration file.

## Models

- The project uses a pre-trained deep learning model for text recognition. You can download the model from the following link: [Model Download](https://example.com).

## Examples

Here's an example of how to use the OCR system:

1. Place an image file (e.g., `sample.jpg`) in the `input_images` folder.
2. Run the OCR script:
   ```bash
   python main.py --input_dir input_images --output_dir output_texts
   ```
3. The output text will be stored in the `output_texts/sample.txt` file.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries, please contact:
- Name: Lê Gia Hưng
- Email: Hunglg.341@gmail.com
