# Housing-Price-Forecast-with-Random-Forest-Regression-
This program focuses on creating a housing price forecasting model using the random forest algorithm and evaluating its performance using various metrics. The program also outputs a visualization of the model's feature importances. 

# Methodologies 
The program uses random forest regression to predict housing prices. Random forest regression is a machine learning model that uses multiple decision trees, each trained on different features and bootstrap samples, to make predictions. These help to make more accurate predictions. Since housing price forecasting is a regression problem, the model outputs the average of the prediction results of all the trees. 

# Model Performance
The model achieves an RMSE (Root Mean Squared Error) of 1,401,263.08 (rounded to 2 decimal places), which means the root of the MSE (Mean Squared Error) from the actual values is 1,401,263.08. Considering this RMSE is lower than the standard deviation of the housing price data, it means that the model can explain the variance. The MAE (Mean Absolute Error) of the model is 1,025,289.68. This indicates that the average absolute loss is 1,025,289.68. The model also achieves a standard R^2 Score of 0.612, which implies that 61.2% of the variance can be explained by the model. This is relatively high. Finally, the average R2 score of the cross-validation check is 0.60, which is also satisfying.

# Feature Importances
Feature Importances show the extent to which each feature variable contributes to the model's performance, and they are sorted in descending order in the following figure. <br>
<img width="1402" height="489" alt="Screenshot 2026-07-26 at 22 35 38" src="https://github.com/user-attachments/assets/6ec404fe-0ae1-4c2e-ade3-6e22b04cc2c2" />

#Regression Line Visualization (Actual Values vs Predicted Values)
<img width="1127" height="478" alt="Screenshot 2026-08-18 at 12 03 11" src="https://github.com/user-attachments/assets/b924c173-98cf-44a0-8a03-f02e6d326bc8" />
The scatter plot above indicates a moderately positive correlation between the actual values and the predicted values of the model, and this corresponds to the R^2 score. The regression visualization reflects the model's limitations, as there are multiple points far from the diagonal, which hinders its predictive capability. 

# Technologies Used
Sklearn<br>
Pandas<br>
Matplotlib<br>

# Dataset & References
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset<br>
https://www.ibm.com/think/topics/random-forest<br>
https://github.com/christinasam/House-Price-Prediction/blob/master/predicting_house_price_with_regression_and_random_forest.ipynb<br>
https://www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/ 
