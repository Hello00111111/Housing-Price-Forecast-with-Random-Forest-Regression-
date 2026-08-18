#importing the packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

#reading the file
data = pd.read_csv('Housing.csv')

#output the summary of the housing price data
description = data['price'].describe()
print('==============================\n')
print('Housing Price Statistics Summary')
print(description)

#defining the dependent and independent variables
X = data.drop('price', axis = 1)
y = data['price']

#encoding the categorical variables
categorical_columns = X.select_dtypes(include = ['object', 'category']).columns.to_list()
le_dict = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

#training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#making predictions
prediction = model.predict(X_test)

#model evaluation metrics 
RMSE = mean_squared_error(y_test, prediction, squared = False)
MAE = mean_absolute_error(y_test, prediction)
r_square = r2_score(y_test, prediction)
cv_score = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'r2')

#feature importances
importances = model.feature_importances_
importance_frame = {'Features': X.columns,
                    'Importance': importances}
importance_frame = pd.DataFrame(importance_frame)
sorted_importance_frame = importance_frame.sort_values(axis = 0, by = 'Importance')

#output the evaluation and feature importances 
print('==============================\n')
print('Evaluation Metrics')
print('Root Mean Squared Error:', RMSE)
print('Mean Absolute Error:', MAE)
print('r^2:', r_square)
print('Average R2 Score using Cross-Validation:', cv_score.mean())
print('==============================\n')
print('Feature Importances')
print(sorted_importance_frame)
print('==============================\n')

#visualize the feature importances
plt.figure(figsize = (15,5))
plt.barh(sorted_importance_frame['Features'], sorted_importance_frame['Importance'], color = 'blue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

#visualize the actual values vs predicted values
plt.figure(figsize = (13,5))
plt.scatter(y_test, prediction)
min_value = min(y_test.min(), prediction.min())
max_value = max(y_test.max(), prediction.max())
plt.plot([min_value,max_value],[min_value,max_value])
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual Values vs Predicted Values')
plt.show()
