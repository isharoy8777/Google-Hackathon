# Google-Hackathon
# 🚀Predicting Combinational Depth in RTL Circuits  
This repository contains a *machine learning model* that predicts the *combinational depth* of RTL (Register-Transfer Level) circuits using a *Random Forest Regressor*. The model is trained on a dataset of circuit features and provides predictions based on signal characteristics.  
## 📂 Project Structure
📁 RTL-Combinational-Depth-Prediction │── 📜 README.md # Project documentation (this file) │── 📜 rtl_dataset.csv # Dataset used for training the model │── 📜 train_model.py # Python script for model training and evaluation │── 📜 predict_depth.py # Script for making predictions on new data │── 📂 images/ # Directory for images and graphs │ ├── feature_importance.png # Graph showing feature importance │ ├── output_example.png # Screenshot of terminal output │── 📜 requirements.txt # Required Python dependencies
## 📊 Dataset  
- The dataset (rtl_dataset.csv) consists of various RTL circuit features.  
- Each row represents a circuit instance, with input features and the target variable *combinational depth*.  

*Sample Data:*  

| fan_in | fan_out | path_length | combinational_depth |
|--------|--------|-------------|----------------------|
| 3      | 5      | 10          | 12                   |
| 4      | 6      | 15          | 14                   |
| 5      | 8      | 20          | 18                   |

📌 Ensure the dataset is in the same directory as the scripts.
## 🛠 Installation & Setup  

1. *Clone the repository*  
   ```bash
   git clone https://github.com/yourusername/rtl-combinational-depth.git  
   cd rtl-combinational-depth
