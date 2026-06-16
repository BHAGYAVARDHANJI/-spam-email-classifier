"""
Spam Email Classifier - Flask Backend API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# ── Global model state ──────────────────────────────────────────────────────
model = None
vectorizer = None
model_metrics = {}


def train_model():
    """Train the Naive Bayes spam classifier."""
    global model, vectorizer, model_metrics

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spam_data.csv')
    df = pd.read_csv(data_path, encoding='latin-1')

    df['label_encoded'] = (df['label'] == 'spam').astype(int)

    vectorizer = CountVectorizer(
        max_features=3000,
        stop_words='english',
        lowercase=True,
        min_df=1,       # relaxed for small dataset
        max_df=0.95
    )

    X = vectorizer.fit_transform(df['message'])
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_test_pred  = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    cm = confusion_matrix(y_test, y_test_pred)

    model_metrics = {
        "train_accuracy":  round(accuracy_score(y_train, y_train_pred), 4),
        "test_accuracy":   round(accuracy_score(y_test,  y_test_pred),  4),
        "precision":       round(precision_score(y_test,  y_test_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_test,    y_test_pred,  zero_division=0), 4),
        "f1_score":        round(f1_score(y_test,        y_test_pred,  zero_division=0), 4),
        "confusion_matrix": cm.tolist(),
        "dataset_size":    len(df),
        "train_size":      X_train.shape[0],
        "test_size":       X_test.shape[0],
        "num_features":    len(vectorizer.get_feature_names_out()),
        "class_distribution": df['label'].value_counts().to_dict(),
    }

    print("✅ Model trained successfully!")
    print(f"   Test Accuracy : {model_metrics['test_accuracy']*100:.2f}%")
    print(f"   F1-Score      : {model_metrics['f1_score']}")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict whether an email is spam or ham."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet"}), 500

    data = request.get_json()
    if not data or 'email' not in data:
        return jsonify({"error": "No email text provided"}), 400

    email_text = data['email'].strip()
    if not email_text:
        return jsonify({"error": "Email text cannot be empty"}), 400

    email_vector = vectorizer.transform([email_text])
    prediction   = model.predict(email_vector)[0]
    proba        = model.predict_proba(email_vector)[0]

    return jsonify({
        "prediction":    "spam" if prediction == 1 else "ham",
        "is_spam":       bool(prediction == 1),
        "spam_prob":     round(float(proba[1]) * 100, 2),
        "ham_prob":      round(float(proba[0]) * 100, 2),
        "email_preview": email_text[:120] + ("..." if len(email_text) > 120 else ""),
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics."""
    if not model_metrics:
        return jsonify({"error": "Model not trained yet"}), 500
    return jsonify(model_metrics)


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple emails at once."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet"}), 500

    data = request.get_json()
    if not data or 'emails' not in data:
        return jsonify({"error": "No emails provided"}), 400

    results = []
    for email_text in data['emails']:
        email_text = email_text.strip()
        if not email_text:
            continue
        vec  = vectorizer.transform([email_text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        results.append({
            "email":      email_text[:80] + ("..." if len(email_text) > 80 else ""),
            "prediction": "spam" if pred == 1 else "ham",
            "is_spam":    bool(pred == 1),
            "spam_prob":  round(float(prob[1]) * 100, 2),
        })

    return jsonify({"results": results, "total": len(results)})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("🚀 Starting Spam Email Classifier API...")
    train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
