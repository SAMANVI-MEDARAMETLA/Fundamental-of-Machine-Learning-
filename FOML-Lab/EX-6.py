
# EX-6: Naive Bayes Classification

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Naive Bayes model
model = GaussianNB()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test)

# Step 7: Display results
print("Actual values:")
print(y_test)

print("\nPredicted values:")
print(y_pred)

# Step 8: Calculate accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Step 9: Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
