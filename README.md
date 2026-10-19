# 🛡️ SMS Shield — SMS Spam Detection System

An intelligent SMS spam detection application built with **Python, Machine Learning, Natural Language Processing (NLP), and Streamlit**. The system analyzes SMS messages, predicts whether they are spam or ham (legitimate), identifies suspicious phrases, and provides risk-awareness information to help users recognize potentially fraudulent messages.

## 📌 Project Overview

SMS spam and scam messages can contain misleading offers, suspicious links, urgent warnings, and requests for sensitive information. SMS Shield helps users analyze message content and identify potential warning signs through machine learning classification and rule-based NLP detection.

The application provides a simple, interactive web interface where users can enter an SMS message and view its predicted category, model confidence when available, detected warning signs, and recommended safety steps.

## ✨ Features

- **SMS Classification:** Predicts whether a message is spam or ham.
- **Machine Learning Models:** Supports Naive Bayes, Logistic Regression, and Linear Support Vector Machine (SVM).
- **NLP-Based Detection:** Identifies suspicious phrases, urgency-related wording, and potentially risky URLs using predefined patterns.
- **Risk Assessment:** Assigns a heuristic risk level of Low, Medium, or High based on detected warning signs.
- **Model Confidence:** Displays spam probability when supported by the selected model.
- **Interactive Streamlit UI:** Provides a simple interface for entering and analyzing messages.
- **Model Training:** Allows users to train the machine learning models from the application.
- **Model Comparison:** Displays saved evaluation results for comparing trained classifiers.
- **Safety Recommendations:** Shows practical precautions for handling suspicious messages.

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| Scikit-learn | Machine learning and model evaluation |
| NLP and Regular Expressions | Text processing and suspicious phrase detection |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web application |
| UCI SMS Spam Collection | SMS classification dataset |

## 📂 Project Structure

```text
SMS_Spam_Detector/
│
├── main.py                 # Streamlit application
├── preprocessing.py        # Text preprocessing
├── nlp_detector.py         # Suspicious phrase and URL detection
├── train_model.py          # Dataset loading, model training and evaluation
├── ai_explainer.py         # Optional Groq AI explanation module
├── sms_spam.csv            # SMS dataset
├── requirements.txt        # Python dependencies
├── .env                    # Local API keys, if needed (do not upload)
├── .gitignore              # Files excluded from Git
│
└── models/
    ├── spam_model.joblib   # Saved best-performing ML model
    └── model_results.joblib # Saved model comparison results
```

*Note: The `models/` directory and its files are generated during training. The optional `ai_explainer.py` module is not used by the current `main.py` interface.*

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

- Dataset source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Ham:** Legitimate, non-spam messages.
- **Spam:** Unwanted or potentially deceptive messages.

The local CSV file should be named `sms_spam.csv` and contain a message column and a label column. Common column names such as `message` and `label`, or `v1` and `v2`, are supported by the dataset-loading code.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/SMS_Spam_Detector.git
cd SMS_Spam_Detector
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.ven
