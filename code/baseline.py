import pandas
from prepare_data import X, y
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

dummy_mean_regressor = DummyRegressor(strategy="mean")
dummy_mean_regressor.fit(X_train, y_train)

score = dummy_mean_regressor.score(X_test, y_test)
print(f"R-squared score of predicting mean was {score}")
