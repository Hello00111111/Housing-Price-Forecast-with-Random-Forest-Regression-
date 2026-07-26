# Housing-Price-Forecast-with-Random-Forest-Regression-
This program focuses on creating a housing price forecasting model using the random forest algorithm and evaluating its performance using various metrics. The program also outputs a visualization of the model's feature importances. 

# Methodologies 
The program uses random forest regression to predict housing prices. Random forest regression is a machine learning model that uses multiple decision trees, each trained on different features and bootstrap samples, to make predictions. Since housing price forecasting is a regression problem, the model outputs the average of the prediction results of all the trees.

# Model Performance
The model achieves an RMSE (Root Mean Squared Error) of 1,401,263.08 (rounded to 2 decimal digits), which means the root of the MSE (Mean Squared Error) from the actual values is 1,401,263.08. Considering this RMSE is lower than the standard deviation of the housing price data, it means that the model can explain the variance. The MAE (Mean Absolute Error) of the model is 1,025,289.68. This indicates that the average absolute loss is 1,025,289.68. The model also achieves a standard R^2 Score of 0.612, which implies that 61.2% of the variance can be explained by the model. This is relatively high. 

# Feature Importances
Feature Importances show to what extent each feature variable contributes to the model's performance, and they are sorted in descending order in the following figure. <br>
<img width="1402" height="489" alt="Screenshot 2026-07-26 at 22 35 38" src="https://github.com/user-attachments/assets/6ec404fe-0ae1-4c2e-ade3-6e22b04cc2c2" />

# Technologies Used
Sklearn<br>
Pandas<br>
Matplotlib<br>

# Dataset & References
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset<br>
https://www.ibm.com/think/topics/random-forest<br>
https://github.com/christinasam/House-Price-Prediction/blob/master/predicting_house_price_with_regression_and_random_forest.ipynb
