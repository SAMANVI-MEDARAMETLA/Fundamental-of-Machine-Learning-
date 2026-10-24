
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# Display results
print("Linear Regression Predictions:", np.round(linear_pred, 2))
print("Polynomial Regression Predictions:", np.round(poly_pred, 2))
print("Linear Regression MSE:", round(mean_squared_error(y, linear_pred), 2))
print("Polynomial Regression MSE:", round(mean_squared_error(y, poly_pred), 2))
print("Linear Regression R2 Score:", round(r2_score(y, linear_pred), 2))
print("Polynomial Regression R2 Score:", round(r2_score(y, poly_pred), 2))

# Plot the results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, linear_pred, label="Linear Regression")
plt.plot(X, poly_pred, label="Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
