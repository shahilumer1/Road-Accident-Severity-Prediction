

# 🚦 Road Accident Severity Prediction

## 📌 Overview

This project predicts the **severity of road accidents** (Slight Injury, Serious Injury, Fatal Injury) using various **Machine Learning algorithms**. The model is trained on accident datasets and deployed as a **Flask web application**, where users can input accident details and get instant predictions.

## 🎯 Objectives

* Analyze accident-related data (driver, vehicle, road, weather conditions).
* Train ML models for **multi-class classification**.
* Deploy a **user-friendly web app** for real-time predictions.

---

## 🛠️ Features

* Supports multiple classifiers:

  * Logistic Regression (Baseline)
  * Decision Tree Classifier
  * Random Forest Classifier
  * Support Vector Machine (SVM)
  * XGBoost Classifier
* Handles categorical features with **encoding**.
* Interactive **Flask-based web interface**.
* Easy to deploy locally or on cloud platforms.

---

## 📂 Project Structure

```
Road-Accident-Severity-Prediction/
│── data/                         # Dataset (CSV files)
│── static/                       # CSS, images, JS
│   └── style.css                 # Web app styling
│── templates/                    # HTML templates
│   └── index.html                # Web app frontend
│── train.py                      # Training script (model building)
│── app.py                        # Flask web application
│── model.pkl                     # Trained ML model
│── encoder.pkl                   # Label encoder for target classes
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Road-Accident-Severity-Prediction.git
cd Road-Accident-Severity-Prediction
```

### 2️⃣ Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Train the Model

```bash
python train.py
```

This will generate:

* `model.pkl` → trained ML model
* `encoder.pkl` → target label encoder

### 2️⃣ Run the Web App

```bash
python app.py
```

Then open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

---

## 📊 Results

* Models compared: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost.
* Best performance achieved with **Random Forest & XGBoost**.
* Accuracy: \~82-85% (depending on dataset split).

---

## 📸 Screenshots

*(Add screenshots of your web app here)*

---

## 🔮 Future Improvements

* Deploy on **Heroku / AWS / Azure**.
* Use **Deep Learning models** for better accuracy.
* Add **real-time accident data APIs** for live predictions.

---

## 🤝 Contributing

Contributions are welcome! Fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

