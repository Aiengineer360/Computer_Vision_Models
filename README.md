# 🧠 Image Classification Projects Collection

This repository brings together multiple image classification projects, each utilizing a unique approach to solve different machine learning or image processing challenges — from heuristic methods with no ML, to traditional machine learning techniques like KNN, SVM, and SGD. These implementations are primarily based on datasets like **MNIST** and **CIFAR-10**, and were developed for academic assignments and competitions.

---

## 📁 Project Overview

### 1. 🔢 Digit Classification using Heuristic Features (No ML)

**Competition**: *S-2025 No ML Digit Classification (Assignment 1)*
**Techniques**: Pure Python + NumPy (No ML or CV libraries)

* **Goal**: Classify handwritten digits (0–9) using simple heuristic features
* **Approach**:

  * Convert grayscale images to binary using thresholding
  * Extract features: **Mean** and **Variance**
  * Use **Nearest Neighbor with Euclidean Distance**
* **Result**: Achieved **86.27% accuracy** on validation set

📂 Files: `train.csv`, `validate.csv`, `test.csv`, `submission.csv`

---

### 2. 🧭 K-Nearest Neighbors (KNN) on CIFAR Dataset

**Competition**: *S-2025 K-Nearest Neighbour (Assignment 2)*
**Libraries**: `numpy`, `Pillow`, `scikit-learn`

* **Goal**: Classify CIFAR images using KNN
* **Key Features**:

  * Convert images to grayscale and flatten
  * Use KNN (`k=3`) with Euclidean distance
* **Result**: Achieved **31% accuracy** on validation set

📂 Structure:

```
knn_assignment/
├── train/    # Training images by class
└── test/     # Test images
```

---

### 3. ⚖️ SVM Linear Classifier on CIFAR-10 (From Scratch)

**Challenge**: Build an SVM classifier without libraries
**Libraries**: `numpy`, `pandas`, `matplotlib` (for visualization only)

* **Goal**: Implement a linear SVM from scratch for CIFAR-10 subset
* **Key Concepts**:

  * Hinge loss + gradient descent
  * Bias term handling
  * Manual training, validation, and test splits
* **Performance**:

  * ✅ Training Accuracy: 100%
  * ⚠️ Validation Accuracy: 30%
  * ❌ Testing Accuracy: 15%

📂 Data: `cifar10_sampled.csv` (187 images across 10 classes)

---

### 4. 💼 Digit Classifier using SGDClassifier (Scikit-learn)

**Technique**: Scikit-learn + SVM (hinge loss)
**Dataset**: MNIST-style CSVs

* **Key Steps**:

  * Normalize pixel values
  * Use `StandardScaler` for feature scaling
  * Train with `SGDClassifier` using `partial_fit`
* **Validation**: Accuracy tracked over 15 epochs
* **Output**: `submission.csv` file with predictions

📂 Files: `train.csv`, `test.csv`, `submission.csv`

---

## 🔧 Requirements

Make sure you have the following packages installed:

```bash
pip install numpy pandas scikit-learn matplotlib pillow
```

---

## 🚀 How to Run

1. Clone the repository.
2. Place any required `.csv` or image datasets in the appropriate folders.
3. Open the respective `.ipynb` notebooks.
4. Run the cells sequentially.
5. Generated predictions will be saved in `submission.csv` or shown via plots.

---

## 📌 Coming Soon

✅ CNN-based models and many more

---

## 👩‍💻 Author

Developed by Arooj — a software engineering student exploring classic and modern approaches to image classification for academic projects and competitions.

---

Let me know if you want it styled in HTML/Markdown for GitHub rendering, or if you'd like a downloadable `.md` version.
