# Fake News Detection System

## Overview

Fake News Detection System is an AI-powered web application that analyzes news articles and determines whether they are **Real News**, **Fake News**, or **Uncertain**.

The system combines:

* Machine Learning (TF-IDF + Logistic Regression)
* Rule-Based Fact Verification
* Google Gemini AI
* FastAPI Backend
* Interactive Frontend (HTML, CSS, JavaScript)

This hybrid approach improves prediction accuracy by combining statistical learning, predefined factual rules, and large language model reasoning.

---

## Features

### AI-Powered News Verification

* Detects Fake News
* Identifies Real News
* Handles Uncertain Claims

### Hybrid Classification Engine

* Rule-Based Verification
* Machine Learning Prediction
* Gemini AI Fact Checking

### Confidence Score

Displays prediction confidence percentage for every result.

### Explanation Generation

Provides a reason behind each prediction.

### Interactive Frontend

* Modern UI
* Responsive Design
* Animated Space-Themed Interface

### REST API Support

FastAPI backend for integration with web and mobile applications.

---

## Technology Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Logistic Regression

### AI Integration

* Google Gemini API

### Utilities

* Joblib
* Python Dotenv
* Pandas
* NumPy

---

## Project Structure

```text
Fake-News-Detection/
│
├── index.html
├── style.css
├── script.js
│
├── ppa.py
├── train_model.py
├── test_api.py
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── .env
├── requirements.txt
│
└── README.md
```

---

## System Architecture

```text
User Input
     │
     ▼
Frontend (HTML/CSS/JS)
     │
     ▼
FastAPI Backend
     │
     ├── Rule-Based Verification
     │
     ├── Gemini AI Fact Checking
     │
     └── Machine Learning Model
                │
                ▼
         Final Prediction
                │
                ▼
     Real / Fake / Uncertain
```

---

## Machine Learning Model

### Text Preprocessing

The dataset undergoes:

* Lowercasing
* URL Removal
* HTML Tag Removal
* Special Character Removal
* Whitespace Normalization

### Feature Extraction

TF-IDF Vectorization

Additional Features:

* Currency Detection
* Number Detection
* Month Detection
* Year Detection
* Weekday Detection
* Percentage Detection

### Algorithm

Logistic Regression with Probability Calibration

---

## API Endpoint

### Home Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Fake News Detection API"
}
```

---

### Prediction Endpoint

```http
POST /predict
```

Request:

```json
{
  "news": "Scientists discovered water on Mars."
}
```

Response:

```json
{
  "prediction": "Real News",
  "confidence": 0.95,
  "reason": "Detected known real event",
  "source": "rule-based"
}
```

---

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
USE_GEMINI=1
```

### Train the Model

```bash
python train_model.py
```

### Run Backend

```bash
uvicorn ppa:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

### Run Frontend

Open:

```text
index.html
```

in your browser.

---

## Example Predictions

### Real News

Input:

```text
The WHO declared COVID-19 a pandemic on March 11, 2020.
```

Output:

```text
Prediction: Real News
Confidence: 95%
```

### Fake News

Input:

```text
Aliens landed in New York and declared war on humans.
```

Output:

```text
Prediction: Fake News
Confidence: 92%
```

---

## Future Enhancements

* BERT-based News Classification
* Real-Time News Source Verification
* News URL Analysis
* User Authentication
* News History Dashboard
* Cloud Deployment
* Multi-Language Support

---

## Learning Outcomes

Through this project:

* Built a full-stack AI application
* Learned FastAPI API development
* Applied NLP techniques
* Implemented Machine Learning models
* Integrated Google Gemini AI
* Developed REST APIs
* Created a responsive frontend

---

## Author

**Naishitha Kandukuri**

B.Tech – Computer Science and Business Systems

SRKR Engineering College

Email: [naishithakandukuri2007@gmail.com](mailto:naishithakandukuri2007@gmail.com)

LinkedIn: [www.linkedin.com/in/naishitha-kandukuri](http://www.linkedin.com/in/naishitha-kandukuri)
