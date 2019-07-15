from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import joblib

from prepare_data import process_data
from utils import root_mean_square_percentage_error
from utils import compare_train_test_error, log_test_results

# Prepare predictors and labels
data = process_data("data/new_train.csv")
X = data.drop(["Sales"], axis=1)
print(X.shape)
print(X.info())
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=False)
# Ensure we are not using future to predict past
assert X_train["Year"].max() <= X_test["Year"].min()


def generate_random_grid():
    # Number of trees in random forest
    n_estimators = [10, 50, 100]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [None, 1, 3, 6, 12, 15, 20, 40, 50, 100]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 20, 100]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8, 16]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    return {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }


def tune_params():
    random_grid = generate_random_grid()
    model = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=model,
                                       param_distributions=random_grid,
                                       cv=3,
                                       n_iter=20,
                                       verbose=2,
                                       n_jobs=-1,
                                       random_state=42)
    random_search.fit(X_train, y_train)
    print("Best params were:\n")
    print(random_search.best_params_)
    best_model = random_search.best_estimator_
    settings = {
        "Model": "RandomForestRegressor",
        "Data": "All cols",
        "Model Description": random_search.best_params_
    }
    error = evaluate(best_model)
    print(f"Root mean squared percentage error: {error}")
    log_test_results(
        settings, error,
        compare_train_test_error(X_train, X_test, y_train, y_test, best_model))


def single_run():
    print("Starting to fit model...")
    params = {
        'n_estimators': 200,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'auto',
        'max_depth': 50,
        'bootstrap': True
    }
    model = RandomForestRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    error = evaluate(model)
    print(f"Root mean squared percentage error: {error}")

    persisted_model_filename = 'models/production.pkl'
    joblib.dump(model, persisted_model_filename)


def evaluate(model):
    return root_mean_square_percentage_error(model.predict(X_test), y_test)


if __name__ == "__main__":
    single_run()
