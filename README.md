# Plant Disease Detection

A deep learning project that detects and classifies plant diseases from leaf images using **Transfer Learning** and **Pretrained VGGNet**. This system helps farmers and agricultural experts quickly identify plant diseases to take timely action.

---

## Features

- Detects multiple plant diseases from leaf images.
- Utilizes **VGGNet pretrained model** for accurate classification.
- Implements **data augmentation** to improve model generalization.
- User-friendly interface for uploading leaf images and getting predictions.
- Can be extended to new plants or disease types.

---

## Dataset

The project uses publicly available plant disease datasets.

- Images are preprocessed and resized to fit the model input.
- Data augmentation techniques applied:
  - Rotation
  - Zoom
  - Horizontal and vertical flips
  - Brightness adjustment

---

## Technology Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Pretrained CNN: VGG16

---

## Installation

 Clone the repository:
   `bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

Usage

Train the model:

python train.py


The model uses VGG16 with transfer learning.

Data augmentation is applied automatically.

Test/predict on new images:

python predict.py --image path_to_image.jpg

Results

Achieves high classification accuracy on standard plant disease datasets.

Example predictions:

Leaf Image	Predicted Disease

	Apple Scab

	Tomato Mosaic Virus
Future Work

Deploy as a web application for real-time predictions.

Add support for more plant species and diseases.

Optimize model for faster inference on mobile devices.
