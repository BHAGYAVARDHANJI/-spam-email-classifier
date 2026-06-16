# 🛡️ Spam Email Classifier — Full Stack ML Project

A full-stack web application that classifies emails as **Spam** or **Ham** using a **Multinomial Naive Bayes** model trained with Scikit-learn, served via a **Flask REST API**, and presented through a modern dark-themed frontend.

---

## 🗂️ Project Structure

```
spam-classifier/
├── backend/
│   └── app.py              # Flask REST API
├── frontend/
│   ├── index.html          # Main UI
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── data/
│   └── spam_data.csv       # Training dataset
├── spam_classifier.py      # Original ML script (standalone)
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- **Single Email Classifier** — paste any email, get instant prediction
- **Batch Classifier** — classify multiple emails at once
- **Model Metrics Dashboard** — live accuracy, precision, recall, F1-score, confusion matrix
- **REST API** — `/api/predict`, `/api/batch-predict`, `/api/metrics`, `/api/health`
- **Multinomial Naive Bayes** with CountVectorizer (bag-of-words)

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask backend
```bash
cd backend
python app.py
```
Server starts at: **http://localhost:5000**

### 5. Open the frontend
Open `frontend/index.html` in your browser — or just visit **http://localhost:5000** (Flask serves it automatically).

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Classify a single email |
| POST | `/api/batch-predict` | Classify multiple emails |
| GET | `/api/metrics` | Get model performance stats |
| GET | `/api/health` | Health check |

### Example — Single Predict
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "WINNER! Claim your FREE prize now!"}'
```

**Response:**
```json
{
  "prediction": "spam",
  "is_spam": true,
  "spam_prob": 94.32,
  "ham_prob": 5.68,
  "email_preview": "WINNER! Claim your FREE prize now!"
}
```

---

## 🤖 ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Multinomial Naive Bayes |
| Feature Extraction | CountVectorizer (Bag of Words) |
| Max Features | 3000 |
| Train / Test Split | 80% / 20% |
| Smoothing (alpha) | 1.0 (Laplace) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn, Pandas, NumPy |
| Backend | Python, Flask, Flask-CORS |
| Frontend | HTML5, CSS3 (custom), Vanilla JS |
| Fonts | Space Grotesk, JetBrains Mono |

---

## 📸 Screenshots

> Add screenshots of your running app here after deployment.

---

## 👤 Author

**Bhagya Vardhan Jaiswal**
