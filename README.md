# garbage-classification-aicte

# 🗑 Garbage Classification Project

### 👨‍💻 Developed by: MANOHAR SAI BARLA  
### 👩‍🏫 Mentor: DULARI BHATT  
### 💼 Internship: AICTE + Shell + Edunet Foundation

---

## 📌 Project Description

This project aims to automate the classification of garbage images using deep learning and transfer learning techniques.  
Proper waste segregation is a growing challenge, and this system helps identify different types of waste (like plastic, paper, metal, etc.) using a trained image classification model.

The project uses a pre-trained EfficientNetB0 model with TensorFlow and Keras. The model was trained, evaluated, and deployed using Gradio for real-time prediction.

---

## 📊 Dataset Used

- *Dataset Source*: [Kaggle - Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)  
- *Classes*: cardboard, glass, metal, paper, plastic, trash  
- *Total Images*: ~2400  
- *Image Size*: Resized to 124x124 pixels for training

---

## 🧠 Model Details

- *Model*: EfficientNetB0 (Transfer Learning)  
- *Input Size*: (124, 124, 3)  
- *Augmentation*: Random rotation, zoom, contrast  
- *Optimizer*: Adam  
- *Loss Function*: SparseCategoricalCrossentropy  
- *Epochs*: 15  
- *Train Accuracy*: ~82.32%  
- *Test Accuracy*: ~81.64%

---

## 🔍 Evaluation

- Plotted training vs validation accuracy and loss  
- Visualized performance using a *confusion matrix*  
- Tested the model through a deployed *Gradio interface*

---

## 🧪 Deployment

The final model was deployed using *Gradio*, allowing users to upload garbage images and receive real-time classification results via a simple web UI.

---

## 📁 Files Included

- week1_garbage_classifier.ipynb – Base code (Week 1)  
- week2_garbage_classifier.ipynb – Improved model with better performance  
- Garbage_Classification_Project.ipynb – Final integrated notebook with outputs  
- README.md – Project overview and documentation
