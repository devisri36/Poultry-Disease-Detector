# 🐔 Poultry Disease Detector

This project is a **machine learning-based image classification system** designed to detect common poultry diseases—Coccidiosis, New Castle Disease (NCD), Salmonella—as well as Healthy birds using deep learning. 
It enables farmers and veterinarians to diagnose diseases quickly by uploading an image of the bird, improving response time and reducing spread.

---

## 📁 Project Structure
Poultry-Disease-Detector/
│
├── app.py # Flask application for disease prediction
├── poultry_model.h5 # Trained CNN model
├── disease.ipynb # Jupyter Notebook used for model training
├── dataset_split/ # Contains training and test image folders
├── index.html # Frontend HTML template
├── uploads/ # Stores uploaded images temporarily
├── static/ # (Optional) Holds raw unsplit dataset
└── .gitignore # Files and folders excluded from Git


---

## 🧠 About the Core Files

- `app.py` - Flask backend that loads the model and serves predictions based on uploaded images.
- `poultry_model.h5` - The trained Keras model saved after fine-tuning with transfer learning.
- `disease.ipynb` - Notebook containing data preprocessing, model training, evaluation, and performance metrics.
- `dataset_split/` - This folder contains the `train/` and `test/` directories with categorized images.

---

## 📦 Dataset

The dataset is manually collected and categorized into the following classes:
- Coccidiosis
- Healthy
- New Castle Disease (NCD)
- Salmonella

📂 **Dataset Download Link:**  
    https://www.kaggle.com/datasets/kausthubkannan/poultry-diseases-detection

---

📄 Project Reports & Templates
- 📎 Full Report (Word): https://drive.google.com/drive/folders/1bewLTMZhPchixh7ZpOs3emGGsPBGFfER?usp=sharing

📎 Technical Documentation:
- Includes architecture, setup instructions, API, authentication, and known issues.

✅ Features
- CNN-based poultry disease classifier

- Web app with simple image upload interface

- Model trained using transfer learning (e.g., VGG16)

- Accuracy over 90% with confusion matrix and performance metrics

⚠️ Notes
- Avoid uploading unclear or heavily shadowed images for accurate results.

- This project supports four categories only; further categories can be added in future versions.

- Ensure PIL/Pillow is installed to use Keras image loaders.

🧪 Future Scope
- Mobile app integration with camera access

- Voice-based or text-based symptom input

- Expanded disease classes

- Multilingual UI for wider rural accessibility
