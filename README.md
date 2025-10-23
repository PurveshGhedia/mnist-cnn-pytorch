# mnist-cnn-pytorch
# 🧠 CNN for MNIST Digit Recognition using PyTorch

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Kaggle](https://img.shields.io/badge/Kaggle-Digit%20Recognizer-success)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)

This repository contains a **Convolutional Neural Network (CNN)** built with **PyTorch** for the **MNIST handwritten digit recognition** task.  
The project is implemented in a **Jupyter Notebook**, demonstrating a complete deep learning workflow — from **data preprocessing** and **model training** to **Kaggle submission generation** for the [Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer).

---

## 🔍 Key Highlights
- Implemented from scratch using **PyTorch**
- Custom **Dataset** and **DataLoader** for MNIST train/test data
- CNN with **Batch Normalization**, **Dropout**, and **ReLU** activation
- Achieved **99.11% validation accuracy (0.99107)** on Kaggle submission
- Generates a ready-to-upload `submission.csv` file for the Kaggle competition

---

## 🚀 Tech Stack
- **Python 3**
- **PyTorch**
- **Pandas**
- **NumPy**
- **Jupyter Notebook**

---

### 🧠 Model Architecture

```markdown
Input: 1 × 28 × 28

Feature Extractor:
├── Conv2d(1, 32, kernel_size=3, padding=1)
├── BatchNorm2d(32)
├── ReLU
├── Conv2d(32, 32, kernel_size=3, padding=1)
├── BatchNorm2d(32)
├── ReLU
├── MaxPool2d(2, 2)
├── Dropout2d(0.2)
│
├── Conv2d(32, 64, kernel_size=3, padding=1)
├── BatchNorm2d(64)
├── ReLU
├── Conv2d(64, 64, kernel_size=3, padding=1)
├── BatchNorm2d(64)
├── ReLU
├── MaxPool2d(2, 2)
├── Dropout2d(0.3)
│
├── Conv2d(64, 128, kernel_size=3, padding=1)
├── BatchNorm2d(128)
├── ReLU
├── Dropout2d(0.3)

Classifier:
├── Flatten
├── Linear(128×7×7 → 512)
├── BatchNorm1d(512)
├── ReLU
├── Dropout(0.3)
│
├── Linear(512 → 256)
├── BatchNorm1d(256)
├── ReLU
├── Dropout(0.3)
│
├── Linear(256 → 10)

```

---

## 📈 Results

| Metric              | Value       |
| :------------------ | :---------- |
| Validation Accuracy | **99.40%**  |
| Kaggle Public Score | **0.99107** |
