import pandas
from prepare_data import X, y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

model = LinearRegression(n_jobs=-1)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"R-squared score: {score}")
