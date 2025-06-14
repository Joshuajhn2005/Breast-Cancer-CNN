# Breast-Cancer-CNN

A CNN-based model to classify breast cancer as benign or malignant using histopathology images.

## 🔬 Project Overview

This project aims to build a Convolutional Neural Network (CNN) called **CancerNet** to classify breast cancer histology images (IDC dataset) as **benign** or **malignant**. The model is trained using Keras and TensorFlow.

## 📁 Dataset

- Dataset: **IDC_regular dataset** from Kaggle
- Total patches: 277,524 (50x50 px)
- Positive (Malignant): 78,786
- Negative (Benign): 198,738

## 🧪 Methodology

1. **Data Preparation**  
   - Organized image patches by labels
   - Applied image preprocessing

2. **Model Architecture (CancerNet)**  
   - Convolutional layers
   - Max-pooling
   - Dropout for regularization
   - Fully connected Dense layers

3. **Training and Evaluation**  
   - Training: 80%, Testing: 20% split
   - Evaluated with Accuracy, Confusion Matrix, and F1-Score

## 📊 Results

| Epochs | Accuracy |
|--------|----------|
| 5      | ~85%     |
| 10     | ~91%     |

- Model showed optimal learning without overfitting.
- Confusion Matrix showed strong true positive and true negative rates.

## 🚀 Real-World Application

If deployed in clinical settings, this model could assist radiologists by providing early insights during diagnosis, potentially saving lives and accelerating treatment decisions.

## 👨‍💻 Tech Stack

- Python
- Keras / TensorFlow
- OpenCV
- NumPy, Matplotlib, Seaborn

## 📎 Source Code

You can find the full source code here:  
🔗 [`CancerNet_BreastCancerClassifier.py`](CancerNet_BreastCancerClassifier.py)

---

Feel free to ⭐ star the repo if you found it useful!
