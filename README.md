<img width="2438" alt="image" src="https://github.com/user-attachments/assets/bdec26aa-dbef-45f7-ab0d-2bdf1cb690ad" />

## 📖 Project Overview

This project implements a **binary image classifier** to distinguish between COVID-19 and normal chest X-rays using PyTorch.  
It replicates a Keras-style deep learning workflow in pure PyTorch, covering image preprocessing, model definition, training, evaluation, and visualization.

## 🎯 Objectives

- Load and preprocess chest X-ray images  
- Convert images to RGB, resize to 224×224, and scale pixel intensities  
- Encode binary labels (`covid19` = 1, `no_covid19` = 0)  
- Build a fully-connected neural network using PyTorch  
- Train the model with SGD and binary cross-entropy loss  
- Evaluate performance using accuracy, loss, and visualizations  

## 🛠 Tools and Libraries

- Python 3.x  
- PyTorch  
- NumPy  
- OpenCV (cv2)  
- Matplotlib  
- imutils  
- scikit-learn  

## 🧠 Model Architecture

- Input: Flattened 224×224×3 image tensor  
- Hidden Layer 1: 300 units + ReLU  
- Hidden Layer 2: 200 units + ReLU  
- Output Layer: 1 unit with sigmoid activation  
- Loss Function: Binary Cross-Entropy (BCE)  
- Optimizer: Stochastic Gradient Descent (SGD)  
- Epochs: 70  

## 📂 Project Structure

covid-xray-pytorch-classifier/  
├── covid_xray_classification_pytorch.ipynb    – Jupyter notebook for training and evaluation  
├── datasets/                                  – Folder with images organized in class subfolders  
│   ├── covid19/                               – Chest X-rays labeled positive  
│   └── no_covid19/                            – Chest X-rays labeled negative  
├── outputs/                                   – (Optional) Folder for plots and saved models  
└── README.md                                  – Project documentation  

## 🧪 Dataset Details

- Format: Images in JPEG or PNG format  
- Folder structure:
  - datasets/covid19/
  - datasets/no_covid19/
- Labels are inferred from the folder name  
- Total samples: user-defined depending on dataset  

## 🚀 How to Run

1. Clone the repository  
2. Place your chest X-ray dataset under `datasets/` with `covid19` and `no_covid19` folders  
3. Open `covid_xray_classification_pytorch.ipynb` in Jupyter Notebook  
4. Run all cells to train and evaluate the model  

## 📊 Example Outputs

- Accuracy and loss plots for training and validation sets  
- Sample chest X-ray visualizations  
- Confusion matrix and classification report on the test set  

## ✅ Requirements

Install all dependencies using:  
pip install torch torchvision numpy opencv-python imutils matplotlib scikit-learn  

## ⚠ Notes

- Ensure all images are readable and formatted consistently  
- Project is built for educational and prototyping purposes  
- Performance will depend on dataset quality and class balance  

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.
