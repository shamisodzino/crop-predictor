import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the data
data = pd.read_csv("crop_data.csv")

# Encode crop names as numeric labels
le = LabelEncoder()
data["Crop"] = le.fit_transform(data["Crop"])

# Split the data into features (X) and target (y)
X = data.drop("Crop", axis=1)
y = data["Crop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the hyperparameter grid
param_grid = {
    "learning_rate": [0.05, 0.1, 0.15],
    "max_depth": [3, 4, 5],
    "n_estimators": [50, 100, 200],
    "reg_lambda": [0.1, 0.5, 1],
    "subsample": [0.5, 0.7, 0.9],
}

# Train an XGBoost Classifier with grid search
clf = XGBClassifier(objective="multi:softmax")
grid_search = GridSearchCV(
    clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring="accuracy"
)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and score
print("Best hyperparameters:", grid_search.best_params_)
print("Best accuracy score:", grid_search.best_score_)

# Save the trained model
clf = grid_search.best_estimator_
with open("xgboost_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Decode numeric labels back to crop names
y_test = le.inverse_transform(y_test)
y_pred = le.inverse_transform(y_pred)

# Evaluate the model
print("Classification report:")
print(classification_report(y_test, y_pred))

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Overall classification accuracy:", accuracy)
