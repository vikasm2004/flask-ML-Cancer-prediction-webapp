## 📌 Project Overview
This project is an end-to-end **Cancer Risk Prediction Web Application** built using **Machine Learning (Random Forest Classifier)** and deployed with **Flask**.  
The system predicts the **cancer risk level** based on multiple health and lifestyle factors provided by the user through a web interface.

The goal of this project is to demonstrate:
- Data preprocessing and feature selection
- Machine learning model training and evaluation
- Model deployment using Flask
- Real-time prediction through a web application

---

## 🧠 Machine Learning Model
- **Algorithm Used:** Random Forest Classifier
- **Reason for Selection:**
  - Handles non-linear relationships effectively
  - Reduces overfitting compared to single decision trees
  - Works well with mixed-feature datasets

---

## 📊 Dataset Information
- **Dataset Name:** `cancer.csv`
- **Target Variable:** `Level`  
  - Represents the cancer risk level (e.g., Low, Medium, High)
- **Selected Features:**
  - Air Pollution
  - Genetic Risk
  - Obesity
  - Balanced Diet
  - Occupational Hazards
  - Coughing of Blood

Feature selection was done to focus on medically relevant and high-impact attributes.

---

## ⚙️ Workflow
1. Load and preprocess the dataset using **Pandas**
2. Select relevant features and target variable
3. Split data into training and testing sets
4. Train a **Random Forest Classifier**
5. Evaluate the model using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
6. Deploy the trained model using **Flask**
7. Accept user input via HTML form
8. Predict cancer risk level in real time

---

## 📈 Model Performance
- The model is evaluated on unseen test data
- Performance metrics include:
  - Accuracy
  - Precision, Recall, and F1-score
  - Confusion Matrix

These metrics help assess the reliability and robustness of the prediction model.

---

## 🌐 Web Application
- **Framework:** Flask
- **Features:**
  - User-friendly input form
  - Real-time predictions
  - Probability distribution across risk levels
- **Endpoints:**
  - `/` → Home page
  - `/predict` → Handles prediction requests

---

## 🛠️ Technologies Used
- Python
- Flask
- Pandas
- Scikit-learn
- HTML/CSS (Frontend)
- Random Forest Algorithm

---

## ▶️ How to Run the Project Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/end-to-end-cancer-risk-prediction-flask-ml.git

---

## 📁 Project Structure
- cancer-risk-prediction-flask-ml/
- │
- ├── app.py # Main Flask application
- ├── cancer.csv # Dataset
- ├── templates/
- │ └── index.html # HTML frontend
- └── README.md # Project documentation
