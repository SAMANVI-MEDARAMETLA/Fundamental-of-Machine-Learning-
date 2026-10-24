
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create sample credit score dataset
data = {
    "Income": [30000, 50000, 70000, 25000, 80000, 60000, 35000, 90000, 45000, 75000],
    "Loan": [10000, 15000, 10000, 20000, 5000, 12000, 18000, 8000, 16000, 7000],
    "Payment_History": [2, 3, 3, 1, 3, 2, 1, 3, 2, 3],
    "Credit_Score": ["Poor", "Standard", "Good", "Poor", "Good",
                     "Standard", "Poor", "Good", "Standard", "Good"]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[["Income", "Loan", "Payment_History"]]
y = df["Credit_Score"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Actual Credit Scores:", list(y_test))
print("Predicted Credit Scores:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Predict credit score for a new person
new_person = pd.DataFrame({
    "Income": [65000],
    "Loan": [10000],
    "Payment_History": [3]
})

print("New Person's Predicted Credit Score:", model.predict(new_person)[0])
