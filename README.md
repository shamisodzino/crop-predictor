# To find the Best performing model, I followed the following strategies:

1. **Tune hyperparameters:** Many Modles have many hyperparameters that can be tuned to improve model performance. Some important hyperparameters to consider include the learning rate, the maximum depth of each tree, the number of trees in the ensemble, and the regularization parameters. You can use techniques like grid search or randomized search to find the best combination of hyperparameters.

2. **Feature engineering:** Feature engineering can involve creating new features from the existing data, scaling or normalizing the features, or removing irrelevant or redundant features. Feature engineering can help the model to better capture patterns in the data and improve its accuracy.

3. **Trying All methods:** Trying a lot of different models out of the box and see which performs the best 

4. **GridSearch:** Find the best parameters for the best methods which included XGBoost , lightgbm , Catboost and RandomForestClassifier.Because of limited resources its impossible to find good hyperparematers for all these models , i ended up just picking a few models that wer perfroming well out of the box and gridsearched them. The grid searched models can be found in the code that end with _hyper i.e train_random_forest_hyper.py

By trying out these strategies, you can improve the accuracy of the model peformance and make it more effective for the task at hand.

### 1. For Random Forests
Classification report:
```
               precision    recall  f1-score   support

       Apples       1.00      0.98      0.99        64
      Bananas       0.97      0.97      0.97        68
     Cabbages       0.75      0.80      0.78        61
       Coffee       0.99      1.00      0.99        71
       Cotton       0.97      0.85      0.90       132
Finger Millet       0.86      1.00      0.93        76
  Ground Nuts       0.71      0.84      0.77        70
        Maize       0.82      0.73      0.77        81
      Oranges       0.97      1.00      0.99        69
 Pearl Millet       0.94      0.85      0.89        68
         Peas       0.97      1.00      0.98        65
     Potatoes       0.98      0.93      0.96        70
      Sorghum       0.74      0.75      0.75        57
     SoyBeans       0.63      0.68      0.65        72
   Sugar Cane       1.00      0.95      0.97        79
          Tea       0.99      0.99      0.99        79
      Tobbaco       0.99      1.00      1.00       101
    Tommatoes       0.83      0.83      0.83        69
        Wheat       0.94      0.91      0.93        68

     accuracy                           0.90      1420
    macro avg       0.90      0.90      0.90      1420
 weighted avg       0.90      0.90      0.90      1420
```
Overall classification accuracy: 0.8992957746478873


### 2. For SVM (Support Vector Machines)
Classification report:
```
               precision    recall  f1-score   support

       Apples       0.96      1.00      0.98        64
      Bananas       0.97      0.94      0.96        68
     Cabbages       0.71      0.69      0.70        61
       Coffee       0.90      1.00      0.95        71
       Cotton       0.73      0.80      0.76       132
Finger Millet       0.76      0.84      0.80        76
  Ground Nuts       0.69      0.60      0.64        70
        Maize       0.70      0.53      0.61        81
      Oranges       0.94      0.99      0.96        69
 Pearl Millet       0.82      0.79      0.81        68
         Peas       0.94      0.95      0.95        65
     Potatoes       0.88      0.83      0.85        70
      Sorghum       0.61      0.65      0.63        57
     SoyBeans       0.44      0.42      0.43        72
   Sugar Cane       0.82      0.89      0.85        79
          Tea       1.00      0.95      0.97        79
      Tobbaco       0.93      0.86      0.89       101
    Tommatoes       0.77      0.83      0.80        69
        Wheat       0.83      0.85      0.84        68

     accuracy                           0.81      1420
    macro avg       0.81      0.81      0.81      1420
 weighted avg       0.81      0.81      0.81      1420
```
Overall classification accuracy: 0.8112676056338028

### 3. For KNN (k-Nearest Neighbors (k-NN):
```
Classification report:
               precision    recall  f1-score   support

       Apples       0.91      0.92      0.91        64
      Bananas       0.98      0.82      0.90        68
     Cabbages       0.52      0.72      0.61        61
       Coffee       0.87      0.93      0.90        71
       Cotton       0.59      0.67      0.63       132
Finger Millet       0.66      0.86      0.75        76
  Ground Nuts       0.64      0.60      0.62        70
        Maize       0.63      0.52      0.57        81
      Oranges       0.89      0.97      0.93        69
 Pearl Millet       0.83      0.66      0.74        68
         Peas       0.75      0.97      0.85        65
     Potatoes       0.68      0.54      0.60        70
      Sorghum       0.46      0.51      0.48        57
     SoyBeans       0.29      0.22      0.25        72
   Sugar Cane       0.81      0.72      0.77        79
          Tea       0.94      0.84      0.89        79
      Tobbaco       0.80      0.98      0.88       101
    Tommatoes       0.59      0.49      0.54        69
        Wheat       0.79      0.54      0.64        68

     accuracy                           0.71      1420
    macro avg       0.72      0.71      0.71      1420
 weighted avg       0.72      0.71      0.71      1420
```
Overall classification accuracy: 0.7140845070422536

### 4. Decision Trees
Classification report:
```
               precision    recall  f1-score   support

       Apples       0.98      0.97      0.98        64
      Bananas       0.88      0.93      0.90        68
     Cabbages       0.76      0.79      0.77        61
       Coffee       0.99      0.97      0.98        71
       Cotton       0.84      0.83      0.84       132
Finger Millet       0.87      0.89      0.88        76
  Ground Nuts       0.73      0.79      0.76        70
        Maize       0.60      0.58      0.59        81
      Oranges       0.94      0.93      0.93        69
 Pearl Millet       0.88      0.82      0.85        68
         Peas       0.97      0.98      0.98        65
     Potatoes       0.88      0.87      0.88        70
      Sorghum       0.58      0.68      0.63        57
     SoyBeans       0.59      0.47      0.52        72
   Sugar Cane       0.85      0.95      0.90        79
          Tea       0.97      0.90      0.93        79
      Tobbaco       0.96      0.93      0.94       101
    Tommatoes       0.84      0.83      0.83        69
        Wheat       0.78      0.84      0.81        68

     accuracy                           0.84      1420
    macro avg       0.84      0.84      0.84      1420
 weighted avg       0.84      0.84      0.84      1420
```
Overall classification accuracy: 0.8401408450704225

### 5. Gradient Boosting Classiffier:

Classification report:
```
               precision    recall  f1-score   support

       Apples       1.00      0.94      0.97        64
      Bananas       0.98      0.96      0.97        68
     Cabbages       0.75      0.82      0.78        61
       Coffee       0.97      0.99      0.98        71
       Cotton       0.89      0.86      0.87       132
Finger Millet       0.88      0.88      0.88        76
  Ground Nuts       0.64      0.87      0.73        70
        Maize       0.79      0.68      0.73        81
      Oranges       0.97      0.97      0.97        69
 Pearl Millet       0.89      0.84      0.86        68
         Peas       0.97      1.00      0.98        65
     Potatoes       0.98      0.90      0.94        70
      Sorghum       0.76      0.67      0.71        57
     SoyBeans       0.65      0.71      0.68        72
   Sugar Cane       0.96      0.96      0.96        79
          Tea       0.99      0.96      0.97        79
      Tobbaco       0.97      0.98      0.98       101
    Tommatoes       0.80      0.81      0.81        69
        Wheat       0.89      0.87      0.88        68

     accuracy                           0.88      1420
    macro avg       0.88      0.88      0.88      1420
 weighted avg       0.88      0.88      0.88      1420
```
Overall classification accuracy: 0.8788732394366198


### 6. XGBoost
Classification report:
```
               precision    recall  f1-score   support

       Apples       0.95      0.97      0.96        64
      Bananas       0.96      0.97      0.96        68
     Cabbages       0.74      0.84      0.78        61
       Coffee       0.97      0.99      0.98        71
       Cotton       0.93      0.88      0.90       132
Finger Millet       0.88      0.88      0.88        76
  Ground Nuts       0.68      0.89      0.77        70
        Maize       0.77      0.68      0.72        81
      Oranges       0.99      0.96      0.97        69
 Pearl Millet       0.94      0.90      0.92        68
         Peas       0.96      1.00      0.98        65
     Potatoes       0.97      0.91      0.94        70
      Sorghum       0.70      0.68      0.69        57
     SoyBeans       0.67      0.64      0.65        72
   Sugar Cane       0.96      0.96      0.96        79
          Tea       0.99      0.97      0.98        79
      Tobbaco       0.97      0.97      0.97       101
    Tommatoes       0.83      0.78      0.81        69
        Wheat       0.84      0.84      0.84        68

     accuracy                           0.88      1420
    macro avg       0.88      0.88      0.88      1420
 weighted avg       0.88      0.88      0.88      1420
```
Overall classification accuracy: 0.8816901408450705
### 7. XGBoost (Grid Search)
i used grid search to try to improve the performance of XGBoost 
Best hyperparameters: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 100, 'reg_lambda': 0.1, 'subsample': 0.7}
Best accuracy score: 0.8834509139341984
Classification report:
```
               precision    recall  f1-score   support

       Apples       1.00      0.97      0.98        64
      Bananas       0.98      0.96      0.97        68
     Cabbages       0.78      0.84      0.81        61
       Coffee       0.99      0.99      0.99        71
       Cotton       0.93      0.87      0.90       132
Finger Millet       0.84      0.99      0.91        76
  Ground Nuts       0.66      0.87      0.75        70
        Maize       0.88      0.72      0.79        81
      Oranges       0.96      0.97      0.96        69
 Pearl Millet       0.94      0.87      0.90        68
         Peas       0.97      1.00      0.98        65
     Potatoes       0.98      0.91      0.95        70
      Sorghum       0.80      0.75      0.77        57
     SoyBeans       0.67      0.62      0.65        72
   Sugar Cane       1.00      0.96      0.98        79
          Tea       0.99      0.97      0.98        79
      Tobbaco       0.96      1.00      0.98       101
    Tommatoes       0.83      0.84      0.83        69
        Wheat       0.87      0.90      0.88        68

     accuracy                           0.90      1420
    macro avg       0.90      0.89      0.89      1420
 weighted avg       0.90      0.90      0.90      1420
```
Overall classification accuracy: 0.8964788732394366

### 8. light Gradient Boost
Classification report:
```
               precision    recall  f1-score   support

       Apples       0.98      0.97      0.98        64
      Bananas       0.96      0.96      0.96        68
     Cabbages       0.76      0.82      0.79        61
       Coffee       0.97      1.00      0.99        71
       Cotton       0.95      0.90      0.93       132
Finger Millet       0.89      0.87      0.88        76
  Ground Nuts       0.68      0.89      0.77        70
        Maize       0.79      0.68      0.73        81
      Oranges       0.96      0.96      0.96        69
 Pearl Millet       0.92      0.85      0.89        68
         Peas       0.97      1.00      0.98        65
     Potatoes       0.96      0.96      0.96        70
      Sorghum       0.69      0.75      0.72        57
     SoyBeans       0.70      0.67      0.68        72
   Sugar Cane       0.97      0.95      0.96        79
          Tea       0.99      0.99      0.99        79
      Tobbaco       0.98      0.98      0.98       101
    Tommatoes       0.85      0.80      0.82        69
        Wheat       0.85      0.85      0.85        68

     accuracy                           0.89      1420
    macro avg       0.89      0.89      0.88      1420
 weighted avg       0.89      0.89      0.89      1420
```
Overall classification accuracy: 0.8887323943661972

### 9.Catboost 
catboost already perfroms gridsearch on its own it was high performing so there was no need to do manual gridsearh on my own 
Classification report:
```
               precision    recall  f1-score   support

       Apples       1.00      0.98      0.99        64
      Bananas       0.94      0.93      0.93        68
     Cabbages       0.73      0.84      0.78        61
       Coffee       0.97      1.00      0.99        71
       Cotton       0.94      0.86      0.90       132
Finger Millet       0.88      0.92      0.90        76
  Ground Nuts       0.72      0.81      0.77        70
        Maize       0.81      0.74      0.77        81
      Oranges       0.94      0.97      0.96        69
 Pearl Millet       0.91      0.93      0.92        68
         Peas       0.96      1.00      0.98        65
     Potatoes       0.95      0.87      0.91        70
      Sorghum       0.71      0.72      0.71        57
     SoyBeans       0.64      0.62      0.63        72
   Sugar Cane       0.95      0.92      0.94        79
          Tea       1.00      0.92      0.96        79
      Tobbaco       0.93      0.98      0.95       101
    Tommatoes       0.85      0.81      0.83        69
        Wheat       0.86      0.88      0.87        68

     accuracy                           0.88      1420
    macro avg       0.88      0.88      0.88      1420
 weighted avg       0.88      0.88      0.88      1420
```
Overall classification accuracy: 0.8816901408450705

### 10. MLC classiffier
Classification report:
```
               precision    recall  f1-score   support

       Apples       0.91      0.95      0.93        64
      Bananas       0.97      0.94      0.96        68
     Cabbages       0.67      0.70      0.69        61
       Coffee       0.96      0.99      0.97        71
       Cotton       0.88      0.81      0.84       132
Finger Millet       0.75      0.84      0.80        76
  Ground Nuts       0.81      0.66      0.72        70
        Maize       0.68      0.72      0.70        81
      Oranges       0.92      0.97      0.94        69
 Pearl Millet       0.82      0.79      0.81        68
         Peas       0.97      0.97      0.97        65
     Potatoes       0.90      0.81      0.86        70
      Sorghum       0.65      0.74      0.69        57
     SoyBeans       0.59      0.57      0.58        72
   Sugar Cane       0.88      0.92      0.90        79
          Tea       1.00      0.97      0.99        79
      Tobbaco       0.94      0.91      0.92       101
    Tommatoes       0.77      0.77      0.77        69
        Wheat       0.79      0.84      0.81        68

     accuracy                           0.84      1420
    macro avg       0.83      0.84      0.83      1420
 weighted avg       0.84      0.84      0.84      1420
```
Overall classification accuracy: 0.8373239436619718
### 11. Light Gradeint Boosting (Grid Search)
i tried grid searching LGB but the i got  a small improvement 
Best parameters: {'learning_rate': 0.1, 'max_depth': 3, 'num_leaves': 31}
Classification report:
```
               precision    recall  f1-score   support

       Apples       1.00      0.95      0.98        64
      Bananas       0.97      0.94      0.96        68
     Cabbages       0.73      0.80      0.77        61
       Coffee       0.97      1.00      0.99        71
       Cotton       0.98      0.89      0.93       132
Finger Millet       0.89      0.92      0.90        76
  Ground Nuts       0.67      0.93      0.78        70
        Maize       0.82      0.73      0.77        81
      Oranges       0.94      0.97      0.96        69
 Pearl Millet       0.92      0.90      0.91        68
         Peas       0.97      0.98      0.98        65
     Potatoes       0.96      0.94      0.95        70
      Sorghum       0.75      0.74      0.74        57
     SoyBeans       0.71      0.64      0.67        72
   Sugar Cane       1.00      0.95      0.97        79
          Tea       0.99      0.99      0.99        79
      Tobbaco       0.98      1.00      0.99       101
    Tommatoes       0.80      0.75      0.78        69
        Wheat       0.87      0.88      0.88        68

     accuracy                           0.89      1420
    macro avg       0.89      0.89      0.89      1420
 weighted avg       0.90      0.89      0.89      1420
```
Overall classification accuracy: 0.893661971830986

### 12. Random Forest (Grid Search) 
random forest waas the best peforming model out of the box so i ended up just doing grid search to find the 
Best parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
Classification report:
```
               precision    recall  f1-score   support

       Apples       1.00      1.00      1.00        64
      Bananas       1.00      0.97      0.99        68
     Cabbages       0.79      0.82      0.81        61
       Coffee       0.99      1.00      0.99        71
       Cotton       0.97      0.85      0.90       132
Finger Millet       0.86      1.00      0.93        76
  Ground Nuts       0.70      0.91      0.80        70
        Maize       0.83      0.73      0.78        81
      Oranges       0.97      1.00      0.99        69
 Pearl Millet       0.94      0.87      0.90        68
         Peas       0.97      1.00      0.98        65
     Potatoes       0.97      0.94      0.96        70
      Sorghum       0.75      0.75      0.75        57
     SoyBeans       0.68      0.67      0.67        72
   Sugar Cane       0.99      0.96      0.97        79
          Tea       0.99      0.99      0.99        79
      Tobbaco       0.99      1.00      1.00       101
    Tommatoes       0.86      0.86      0.86        69
        Wheat       0.95      0.91      0.93        68

     accuracy                           0.91      1420
    macro avg       0.91      0.91      0.90      1420
 weighted avg       0.91      0.91      0.91      1420
```
Overall classification accuracy: 0.9070422535211268

# Here are the steps to run the Streamlit app on Windows and Linux using a virtual environment:

1. **Install Python:** Make sure you have Python installed on your system. If you don't have it installed, you can download it from the official Python website: https://www.python.org/downloads/

2. **Create a virtual environment:** : Open a terminal (Command Prompt for Windows or Terminal for Linux) and navigate to the directory where you want to create your project. Run the following command to create a virtual environment named "venv":

For Windows:

```
python -m venv venv
```

For Linux:

```
python3 -m venv venv
```

3. **Activate the virtual environment:**  To activate the virtual environment, run the following command:

For Windows:

```
.\venv\Scripts\activate
```
For Linux:
```
source venv/bin/activate
```
Your prompt should now show the name of the virtual environment, indicating that it's activated.

4. **Install Streamlit and other required libraries:** Create a requirements.txt file in your project folder with the following contents:

Run the following command to install the required libraries from the requirements.txt file:

```
pip install -r requirements.txt
```

5. **Run Streamlit app:** To run the app, execute the following command in the terminal:
```
streamlit run app.py
``` 
Open the app in a web browser: Streamlit will provide a URL in the terminal output, e.g., http://localhost:8501. Open this URL in your web browser to view and interact with the app.

To stop the app, press Ctrl+C in the terminal. To deactivate the virtual environment, simply type deactivate and press Enter .