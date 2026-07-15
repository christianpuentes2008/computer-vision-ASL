import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "asl_landmarks.csv"
MODEL_FILE = "asl_model.joblib"

# Load data
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} samples across {df['label'].nunique()} letters")
print(df['label'].value_counts().sort_index())

# Split features (landmark coords) and labels (letters)
X = df.drop("label", axis=1)
y = df["label"]

# Train/test split so we can measure accuracy honestly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Random Forest classifier - fast, accurate, no GPU needed
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest accuracy: {accuracy:.2%}\n")
print(classification_report(y_test, predictions))

# Save the trained model to disk
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")