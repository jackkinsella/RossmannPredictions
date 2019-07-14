from sklearn.ensemble import RandomForestRegressor
from prepare_data import X, y
from utils import root_mean_square_percentage_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False)
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X, y)

error = root_mean_square_percentage_error(model.predict(X_test), y_test)
print(f"Root mean squared percentage error: {error}")
