import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the data
data = pd.read_csv("crop_data.csv")

# Split the data into features (X) and target (y)
X = data.drop("Crop", axis=1)
y = data["Crop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LGBM Classifier
clf = LGBMClassifier(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 63, 127],
}

# Perform grid search
grid = GridSearchCV(clf, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters:", grid.best_params_)

# Save the best model
best_model = grid.best_estimator_
with open("lightgbm_classifier_best.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Classification report:")
print(classification_report(y_test, y_pred))

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Overall classification accuracy:", accuracy)
