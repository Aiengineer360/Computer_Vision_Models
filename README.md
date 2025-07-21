# 🧠 Image Classification Projects Collection

This repository brings together multiple image classification projects, each utilizing a unique approach to solve different machine learning or computer vision challenges - from heuristic methods with no ML, to traditional machine learning techniques, deep learning architectures, and transfer learning solutions.

---

## 📁 Project Overview

### 1. 🔢 Digit Classification using Heuristic Features (No ML)

**Competition**: *S-2025 No ML Digit Classification (Assignment 1)*  
**Techniques**: Pure Python + NumPy (No ML or CV libraries)  

* **Goal**: Classify handwritten digits (0-9) using simple heuristic features  
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

### 5. 🧠 Feed Forward Neural Network for CIFAR-10

**Challenge**: *S-2025 Basic Neural Network Classification*  
**Libraries**: `tensorflow`, `keras`, `numpy`  

* **Goal**: Classify CIFAR-10 images using raw pixels with FFNN  
* **Requirements**:
  * No preprocessing/feature extraction  
  * No dropouts, batch norm, or data augmentation  
* **Implementation**:
  * One-hot encoded labels  
  * 3 hidden layers (512, 256, 128 units) with ReLU  
  * 50 epochs training  
* **Result**: **55% accuracy** (Ranked #5 on leaderboard)

<img width="1014" height="352" alt="image" src="https://github.com/user-attachments/assets/e0f72c2a-29be-4905-8e9e-a93d2b53b0c1" />

---

### 6. 🖼️ Custom CNN for CIFAR-10 Classification

**Challenge**: *S-2025 CNN Image Classification*  
**Libraries**: `tensorflow`, `keras`, `matplotlib`  

* **Goal**: Build custom CNN for 32x32x3 images  
* **Key Features**:
  * Custom CNN architecture (Conv->Pool->Conv->Pool->Dense)  
  * Training/validation accuracy/loss visualization  
  * Allowed: Dropout, BatchNorm, Data Augmentation  
* **Performance**:
  * **81% test accuracy**  
  * Ranked #2 on competition leaderboard
  * 
<img width="1030" height="284" alt="image" src="https://github.com/user-attachments/assets/f74373d3-99c3-4b64-b926-43e1ccedd9e2" />

---

### 7. 🐾 ResNet for 164x164x3 Image Classification (11 Classes)

**Challenge**: *S-2025 Advanced Image Classification*  
**Libraries**: `tensorflow`, `keras`  

* **Dataset**: Custom 11-class dataset (164x164x3 images)  
* **Approach**:
  * Implemented ResNet architecture  
  * 50 epochs training  
  * Class labels 0-10  
* **Key Techniques**:
  * Skip connections  
  * Batch normalization  
  * Data augmentation  
* **Result**: **89% validation accuracy**  

📂 Structure:
```
resnet_project/
├── data/          # Training images
├── predictions/   # Test results
└── resnet11.h5    # Saved model
```

---

### 8. 🎭 Transfer Learning for Face Classification (16 Classes)

**Challenge**: *S-2025 Face Recognition Competition*  
**Libraries**: `torch`, `torchvision`, `sklearn`  

* **Special Requirements**:
  * Class -1: "Unknown person" category  
  * Multi-metric evaluation (Top-1, Top-3, F1)  
* **Solution**:
  * ResNet50 base with custom head  
  * Fine-tuned last 5 layers  
  * Class-weighted loss function  
* **Performance**:
  * **96% Top-1 accuracy**  
  * Ranked #3 on leaderboard  
  * F1-score: 0.95  
* **Output**: `face_classifier.pth`  

---

### 9. 🚀 YOLOv8 for Gender Detection

**Custom Project**: Gender classification from images  
**Tools**: `ultralytics`, `roboflow`  

* **Workflow**:
  1. Dataset labeling in RoboFlow  
  2. YOLOv8 model training  
  3. Validation and testing  
* **Key Metrics**:
  * Precision: 92%  
  * Recall: 89%  
  * mAP@0.5: 0.91  
* **Usage**:
  ```python
  model = YOLO('gender_detection.pt')
  results = model.predict(source='input.jpg')
  ```
---

### 10. 👥 Complete Face & Gender Detection Pipeline

**Integration Project**: Combines Models #8 and #9  
**Components**:
1. Face detection (MTCNN)  
2. Gender classification (YOLOv8)  
3. Identity recognition (ResNet classifier)  

* **Pipeline Flow**:
  ```
  Input Image → Face Detection → Gender Prediction → Identity Classification → Output
  ```
* **Performance**:
  - End-to-end accuracy: 94%  
  - Processing speed: 12 FPS (1080p)
  
<img width="1012" height="290" alt="image" src="https://github.com/user-attachments/assets/332ec677-e60c-4f29-8914-2b80da951a6e" />

📂 Structure:
```
pipeline/
├── face_detector/  
├── gender_model/  
├── identity_classifier/  
└── pipeline.ipynb  # Complete implementation
```

---

## 🔧 Requirements

```bash
pip install numpy pandas scikit-learn matplotlib pillow
pip install tensorflow torch torchvision ultralytics
```

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies (see above)
3. For each project:
   - Place datasets in specified folders
   - Open the corresponding notebook
   - Run cells sequentially
4. Models will:
   - Save weights (.h5/.pth/.pt)
   - Generate predictions
   - Create visualizations

---

## 📊 Performance Summary

| Project | Model Type | Accuracy |
|---------|------------|----------|
| 1 | Heuristic | 86.27% |
| 2 | KNN | 31% |
| 3 | SVM (scratch) | 15% |
| 4 | SGDClassifier | 85% |
| 5 | FFNN | 55% |
| 6 | Custom CNN | 81% |
| 7 | ResNet | 89% |
| 8 | Transfer Learning | 96% |
| 9 | YOLOv8 | 92% P |
| 10 | Pipeline | 94% |
---

## 👩‍💻 Author

Developed by **Arooj** - a software engineering student exploring classic and modern approaches to image classification through academic projects and competitions.

```
