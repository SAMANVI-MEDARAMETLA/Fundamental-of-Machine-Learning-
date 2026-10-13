
# EX-8: Linear Regression

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Step 3: Create Linear Regression model
model = LinearRegression()

# Step 4: Train model
model.fit(X, y)

# Step 5: Predict values
y_pred = model.predict(X)

# Step 6: Display results
print("Actual values:", y.ravel() if y.ndim > 1 else y)
print("Predicted values:", y_pred)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Step 7: Calculate evaluation metrics
print("Mean Squared Error:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

# Step 8: Predict a new value
new_value = model.predict([[6]])
print("Prediction for X = 6:", new_value[0])

# Step 9: Plot regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()
