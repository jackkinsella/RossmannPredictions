import pandas
from prepare_data import X, y
from utils import root_mean_square_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False)

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
error = root_mean_square_percentage_error(dummy.predict(X_test), y_test)
print("Baseline (Mean):")
print(f"Dummy Root mean squared percentage error: {error}")

print("\nLinear Regression:")
model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)
error = root_mean_square_percentage_error(model.predict(X_test), y_test)
print(f"Root mean squared percentage error: {error}")
