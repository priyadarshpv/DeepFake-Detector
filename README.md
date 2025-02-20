# DeepFake-Detector 🔍

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning-based application to detect deepfake images using **Xception** and **Streamlit**. This project aims to identify synthetic media and ensure authenticity in digital content.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Deepfake technology has become increasingly sophisticated, making it harder to distinguish between real and fake images. This project leverages a **deep learning model** (Xception) trained on a dataset of real and fake images to classify uploaded images as "Real" or "Fake."

The app is built using **Streamlit**, making it easy to use and deploy.

---

## Features
- **Image Upload**: Upload an image to classify it as "Real" or "Fake."
- **Real-Time Prediction**: Get instant results with confidence scores.
- **User-Friendly Interface**: Simple and intuitive UI powered by Streamlit.
- **Scalable**: Built with TensorFlow, allowing for easy model upgrades.

---

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- PIL (Pillow)
- NumPy

### **Preview**

Generated Fake images:

![Screenshot 2025-02-20 171908](https://github.com/user-attachments/assets/098f58d6-e75d-41b6-9b43-8bfde99aa3cd)
![Screenshot 2025-02-20 172105](https://github.com/user-attachments/assets/4ea0dcc6-c780-46a7-acb9-e7b95260b53d)

Real images:

![Screenshot 2025-02-20 175819](https://github.com/user-attachments/assets/de6fc1b4-b9db-40d0-84bb-9f805d3381af)
![Screenshot 2025-02-20 175937](https://github.com/user-attachments/assets/e3d80993-4b66-47ed-8b14-3d2cb164c0e2)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DeepFake-Detector.git
   cd DeepFake-Detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained model:
   - Place the `xception_deepfake_detection_model.h5` file in the root directory.
   - (Optional) Train your own model using the provided dataset.

---

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`.

3. Upload an image using the file uploader.

4. View the prediction result ("Real" or "Fake") along with the confidence score.

---

## Dataset
The model was trained on the [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) from Kaggle. 
The dataset contains:
- **Real Images**: Authentic images of human faces.
- **Fake Images**: Deepfake-generated images.

---

## Model Architecture
The model is based on **Xception**, a powerful convolutional neural network (CNN) architecture. Key components:
- **Base Model**: Xception (pretrained on ImageNet).
- **Custom Layers**:
  - Global Average Pooling
  - Dense Layers with Dropout
  - Sigmoid Activation for Binary Classification

### Training Details
- **Input Size**: 224x224
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Epochs**: 10 (initial training) + 5 (fine-tuning)

---

## Contributing
Contributions are welcome! Here’s how you can help:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please ensure your code follows the project's style and includes appropriate tests.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- Kaggle for the [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images).
- TensorFlow and Streamlit for their amazing libraries.
- The open-source community for continuous inspiration.

---

## Contact
For questions or feedback, feel free to reach out:
- **Your Name**  
- **Email**: your.email@example.com  
- **GitHub**: [yourusername](https://github.com/yourusername)  



### **How to Use This README**
1. Copy the content above into a file named `README.md` in your repository.
2. Replace placeholders like `yourusername`, `your.email@example.com`, and `LICENSE` with your actual details.
3. Add a `requirements.txt` file with the necessary dependencies:
   ```plaintext
   tensorflow==2.x
   streamlit==1.x
   numpy==1.x
   pillow==10.x
   ```
4. Push the changes to GitHub.

---


