from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from prepare_data import X, y, settings
from utils import root_mean_square_percentage_error, compare_train_test_error, log_test_results
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False)
# Ensure we are not using future to predict past
assert X_train["Year"].max() <= X_test["Year"].min()

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
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=model, param_distributions=random_grid, cv=3, n_iter=20, verbose=2, n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)


def evaluate(model):
    return root_mean_square_percentage_error(model.predict(X_test), y_test)


print("Best params were:\n")
print(random_search.best_params_)
best_model = random_search.best_estimator_
settings.update(
    {
        "Model": "RandomForestRegressor",
        "Model Description": random_search.best_params_
    }
)
error = evaluate(best_model)
print(f"Root mean squared percentage error: {error}")
log_test_results(settings, error, compare_train_test_error(X_train, X_test,
                                                           y_train, y_test, best_model))
