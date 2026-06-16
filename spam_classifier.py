"""
Spam Email Classifier using Naive Bayes
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("SPAM EMAIL CLASSIFIER - MACHINE LEARNING PROJECT")
print("=" * 70)
print("\n[STEP 1] Loading and exploring the dataset...\n")

# Load data
df = pd.read_csv('spam_data.csv', encoding='latin-1')

print(f"Dataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print("\n" + "=" * 70)
print("[STEP 2] Data Preprocessing...")
print("=" * 70 + "\n")

df['label_encoded'] = (df['label'] == 'spam').astype(int)

print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

print("\n" + "=" * 70)
print("[STEP 3] Feature Extraction (CountVectorizer)...")
print("=" * 70 + "\n")

vectorizer = CountVectorizer(
    max_features=3000,
    stop_words='english',
    lowercase=True,
    min_df=2,
    max_df=0.95
)

X = vectorizer.fit_transform(df['message'])
y = df['label_encoded']

print(f"Feature Matrix Shape: {X.shape}")
print(f"Number of features: {len(vectorizer.get_feature_names_out())}")

print("\n" + "=" * 70)
print("[STEP 4] Splitting data into training and testing sets...")
print("=" * 70 + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

print("\n" + "=" * 70)
print("[STEP 5] Training Multinomial Naive Bayes Model...")
print("=" * 70 + "\n")

nb_classifier = MultinomialNB(alpha=1.0)
nb_classifier.fit(X_train, y_train)

print("Model training completed!")

print("\n" + "=" * 70)
print("[STEP 6] Model Evaluation...")
print("=" * 70 + "\n")

y_train_pred = nb_classifier.predict(X_train)
y_test_pred = nb_classifier.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("TRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_train, y_train_pred):.4f}")

print("\nTESTING SET PERFORMANCE:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"  F1-Score:  {f1_score(y_test, y_test_pred):.4f}")

cm = confusion_matrix(y_test, y_test_pred)
print("\nCONFUSION MATRIX:")
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

print("\n" + "=" * 70)
print("[STEP 7] Creating visualizations...")
print("=" * 70 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False)
ax1.set_title('Confusion Matrix')

ax2 = axes[0, 1]
accuracies = [train_accuracy, test_accuracy]
ax2.bar(['Training', 'Testing'], accuracies, color=['#2ecc71', '#3498db'])
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Comparison')
ax2.set_ylim([0.9, 1.0])

ax3 = axes[1, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [test_accuracy, precision_score(y_test, y_test_pred), 
          recall_score(y_test, y_test_pred), f1_score(y_test, y_test_pred)]
ax3.barh(metrics, values, color='#e74c3c')
ax3.set_title('Performance Metrics')

ax4 = axes[1, 1]
class_counts = df['label'].value_counts()
ax4.pie(class_counts, labels=['Ham', 'Spam'], autopct='%1.1f%%')
ax4.set_title('Class Distribution')

plt.tight_layout()
plt.savefig('spam_classifier_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'spam_classifier_results.png'")
plt.show()

print("\n" + "=" * 70)
print("[STEP 8] Testing with sample emails...")
print("=" * 70 + "\n")

sample_emails = [
    "Hi, how are you doing today?",
    "WINNER!!! You have won a FREE prize! Click here to claim!",
    "Let's meet tomorrow at the coffee shop",
    "Congratulations! You've won £1000. Claim now!",
]

for i, email in enumerate(sample_emails, 1):
    email_vector = vectorizer.transform([email])
    prediction = nb_classifier.predict(email_vector)[0]
    confidence = nb_classifier.predict_proba(email_vector)[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    spam_prob = confidence[1] * 100
    
    print(f"{i}. Email: \"{email}\"")
    print(f"   Prediction: {label} (Spam: {spam_prob:.2f}%)\n")

print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)