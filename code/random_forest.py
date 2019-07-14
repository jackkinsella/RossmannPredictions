from sklearn.ensemble import RandomForestRegressor

from prepare_data import X, y
from utils import root_mean_square_percentage_error, compare_train_test_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False)
# Ensure we are not using future to predict past
assert X_train["Year"].max() <= X_test["Year"].min()

model = RandomForestRegressor(
    n_estimators=10, max_depth=1000, random_state=42, n_jobs=-1)
model.fit(X, y)

error = root_mean_square_percentage_error(model.predict(X_test), y_test)
print(f"Root mean squared percentage error: {error}")

print(

    compare_train_test_error(X_train, X_test, y_train, y_test, model)
)
