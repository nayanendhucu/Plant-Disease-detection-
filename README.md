#  Plant Disease Detection Using Deep Learning

A deep learning-based image classification project that detects plant diseases from leaf images using **Transfer Learning with MobileNetV2**. The project uses the PlantVillage dataset and follows an end-to-end computer vision workflow including dataset preparation, image preprocessing, data augmentation, model training, validation, model saving, and prediction on new leaf images.

## 📌 Project Overview

Plant diseases can significantly affect crop health and agricultural productivity. This project explores how deep learning and image processing can be used to automatically identify plant diseases from leaf images.

A **pre-trained MobileNetV2 model**, originally trained on ImageNet, is used as the feature-extraction backbone. The convolutional base is frozen and a custom classification head is added to classify the input image into one of **16 plant disease/health classes**.

The trained model can also be used to analyze new leaf images and return the predicted disease class along with a confidence score.



## 🎯 Objectives

* Detect plant diseases from leaf images.
* Apply image preprocessing and augmentation techniques.
* Organize image data into training, validation, and testing sets.
* Implement transfer learning using MobileNetV2.
* Train a multi-class image classification model.
* Evaluate model performance using validation accuracy.
* Save the trained model for later inference.
* Predict diseases from newly uploaded leaf images.


## 🗂️ Dataset

The project uses a **PlantVillage-based plant leaf image dataset** obtained through Kaggle.

The notebook organizes the images into:

* **Training:** 53,693 images
* **Validation:** 12,067 images
* **Testing:** 1,358 images
* **Total:** 67,118 images
* **Classes:** 16

### Kaggle Dataset

**Plant Village Dataset — Kaggle**

[Plant Village Dataset on Kaggle](https://www.kaggle.com/datasets/madhavanl/plant-village-dataset?utm_source=chatgpt.com)

The PlantVillage dataset is a widely used dataset for image-based plant disease research. The original PlantVillage collection contains healthy and diseased plant leaf images organized by plant/disease categories.

> **Note:** The exact Kaggle source URL is not stored inside the uploaded notebook, so the link above should be treated as the referenced PlantVillage Kaggle dataset rather than a claim that this exact Kaggle page was recorded in the notebook.



## 🌱 Classes

The notebook contains 16 classification classes:

1. Apple Scab
2. Bacterial Spot
3. Black Rot
4. Cedar Apple Rust
5. Cercospora Leaf Spot
6. Common Rust
7. Early Blight
8. Esca (Black Measles)
9. Healthy
10. Late Blight
11. Leaf Blight
12. Leaf Scorch
13. Northern Leaf Blight
14. Powdery Mildew
15. Septoria Leaf Spot
16. Yellow Leaf Curl Virus



## 🔄 Project Workflow

```text
PlantVillage Dataset
        ↓
Dataset Extraction
        ↓
Dataset Organization
        ↓
Train / Validation / Test Split
        ↓
Image Preprocessing
        ↓
Data Augmentation
        ↓
MobileNetV2 Transfer Learning
        ↓
Custom Classification Head
        ↓
Model Training
        ↓
Validation
        ↓
Save Trained Model
        ↓
Upload New Leaf Image
        ↓
Disease Prediction + Confidence
```

---

## 🧹 Image Processing & Data Preparation

The notebook first extracts the dataset and organizes the plant images into separate training, validation, and testing directories.

Images are resized to:

**224 × 224 pixels**

Pixel values are normalized using:

```python
rescale = 1./255
```

Training images are augmented using:

* Rotation
* Zoom
* Horizontal flipping

Validation and test images are only rescaled without augmentation.

This allows the model to encounter variations in training images while keeping validation and test processing consistent.



## 🧠 Model Architecture

The project uses **MobileNetV2 with ImageNet pre-trained weights** as the base model.

The MobileNetV2 convolutional layers are frozen and a custom classification head is added:

```text
Input Image
224 × 224 × 3
       ↓
MobileNetV2
ImageNet Pre-trained
       ↓
Global Average Pooling
       ↓
Dropout (30%)
       ↓
Dense Layer
128 Neurons + ReLU
       ↓
Softmax Output
16 Classes
```

### Why MobileNetV2?

MobileNetV2 is a lightweight convolutional neural network architecture designed to provide efficient image feature extraction with relatively low computational requirements. In this project, its pre-trained ImageNet representations are reused through transfer learning.

---

## ⚙️ Training Configuration

| Parameter            | Value                    |
| -------------------- | ------------------------ |
| Model                | MobileNetV2              |
| Pre-trained Weights  | ImageNet                 |
| Input Size           | 224 × 224 × 3            |
| Batch Size           | 32                       |
| Optimizer            | Adam                     |
| Loss Function        | Categorical Crossentropy |
| Output Activation    | Softmax                  |
| Dropout              | 0.3                      |
| Dense Layer          | 128 neurons              |
| Training Epochs      | 2                        |
| Training Steps/Epoch | 200                      |
| Validation Steps     | 50                       |

---

## 📊 Model Performance

The model was trained for 2 epochs.

| Metric                              |     Result |
| ----------------------------------- | ---------: |
| Epoch 1 Training Accuracy           |     55.96% |
| Epoch 1 Validation Accuracy         |     81.50% |
| Epoch 2 Training Accuracy           |     84.52% |
| Epoch 2 Validation Accuracy         |     86.94% |
| Final Evaluated Validation Accuracy | **87.52%** |

The final validation evaluation produced an accuracy of approximately **87.52%**.

> The reported performance is validation performance from this notebook. It should not be interpreted as real-world agricultural diagnostic accuracy.

---

## 🔍 Prediction on New Images

After training, the model is saved as:

```text
plant_disease_model.h5
```

New leaf images can then be uploaded and processed by the trained model.

The inference pipeline:

```text
New Leaf Image
      ↓
Resize to 224 × 224
      ↓
Normalize Pixel Values
      ↓
Model Prediction
      ↓
Find Highest Probability Class
      ↓
Predicted Disease
      ↓
Confidence Score
```

The notebook demonstrates prediction on uploaded images such as:

* `healthyappleleaf.jpg`
* `potato e blight leaf.jpg`
* `tomatoleaf.jpg`

For example, the notebook predicted **Healthy** for `healthyappleleaf.jpg` with approximately **99.55% confidence**.

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **MobileNetV2**
* **NumPy**
* **Matplotlib**
* **Google Colab**
* **Kaggle Dataset**
* **ImageDataGenerator**
* **Transfer Learning**

---

## 📁 Project Structure

A suggested GitHub structure:

```text
Plant-Disease-Detection/
│
├── Plant_Disease_Detection.ipynb
├── plant_disease_model.h5
├── README.md
└── images/
    └── sample_predictions/
```

The full dataset should **not** be uploaded to GitHub because of its size. Instead, provide the Kaggle dataset link in the README.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Open the notebook

Open:

```text
Plant_Disease_Detection.ipynb
```

using Google Colab or Jupyter Notebook.

### 3. Download the dataset

Download the PlantVillage dataset from Kaggle and upload/extract it into the Colab environment according to the directory structure expected by the notebook.

### 4. Install dependencies

```bash
pip install tensorflow numpy matplotlib
```

### 5. Run the notebook

Execute the cells sequentially to:

* Extract the dataset
* Organize the images
* Create image generators
* Apply preprocessing and augmentation
* Build the MobileNetV2 model
* Train the classifier
* Evaluate validation performance
* Save the model
* Predict new leaf images

---

## 📌 Key Learning Outcomes

Through this project, I gained practical experience in:

* Computer vision and image classification
* Image preprocessing
* Image augmentation
* Multi-class classification
* Deep learning with TensorFlow/Keras
* Transfer learning
* MobileNetV2 architecture
* Model evaluation
* Model serialization
* Prediction on unseen images
* Working with large image datasets in Google Colab

---

## ⚠️ Limitations

The model was trained and evaluated using the dataset available in this project. PlantVillage-style datasets primarily contain controlled leaf images, so performance on real-world photographs with different lighting, backgrounds, camera quality, or multiple leaves may differ.

The model is intended as a **machine learning/educational project** and should not be treated as a definitive agricultural diagnosis system.

---

## 📚 Dataset Reference

The PlantVillage dataset is a well-known research dataset for plant disease image classification and was introduced in research on deep learning-based plant disease detection.

**Dataset:** PlantVillage
**Source:** Kaggle
**Task:** Multi-class plant disease image classification
