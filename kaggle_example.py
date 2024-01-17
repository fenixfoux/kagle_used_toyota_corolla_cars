"""
THIS IS AN EXAMPLE FROM KAGGLE.COM https://www.kaggle.com/code/vishakhdapat/used-toyota-corolla-cars-price-prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('ToyotaCorolla.csv')
# print(df.head())

# Check Null Values
# print(df.isnull().sum())

# Check datatypes of Variables
# print(df.info())

# Show basic summary statistic
# print(df.describe())

# Drop 'ID' and 'Model'.

df1 = df.drop(columns= ['Id', 'Model'], axis=1)
# print(df1.head()) # 5 rows × 37 columns

# Convert Fuel_type from categorical to numerical
unique_values = df1['Fuel_Type'].unique()
mapping_dict = {value: index + 1 for index, value in enumerate(unique_values)}

df1['Fuel_Type_numeric'] = df1['Fuel_Type'].map(mapping_dict)
# print(df1[['Fuel_Type_numeric', 'Fuel_Type']].head(50)) # 50 rows × 38 columns


# Convert Color from categorical to numerical
unique_values = df1['Color'].unique()
mapping_dict = {value: index + 1 for index, value in enumerate(unique_values)}

df1['Color_numeric'] = df1['Color'].map(mapping_dict)
# print(df1[['Color_numeric', 'Color']].head(50)) # 50 rows × 39 columns

# Drop extra columns
df2 = df1.drop(columns= ['Fuel_Type', 'Color'])
# print(len(df2.columns)) # 37 columns

# Store Independent variables in X and Dependent Variable in y
X = df2.drop(columns=['Price'], axis=1)
y = df2['Price']
# print(y.head())

# Check Multicollinearity Assumption
"""
One important problem in developing multiple regression models involves the possible collinearity of the independent 
variables. This condition refers to situations in which two or more of the independent variables are highly correlated 
with each other. In such situations, collinear variables do not provide unique information, and it becomes difficult to 
separate the effects of such variables on the dependent variable. When collinearity exists, the values of the regression
coefficients for the correlated variables may fluctuate drastically. One method of measuring collinearity is to 
determine the variance inflationary factor (VIF) for each independent variable.
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF for {X.columns[i]} : {vif}")

"""
Above output suggests that the variance inflation factor (VIF) for the independent variables in dataset is infinite 
(inf). This situation typically arises when there is perfect multicollinearity among the variables. Perfect 
multicollinearity occurs when one or more independent variables in a regression model can be exactly predicted by a 
linear combination of the other variables.

In this case, the VIF is calculated for the variables "Age_08_04", "Mfg_Month", and "Mfg_Year", and the result is 
infinite. This indicates that these variables are perfectly correlated with the other variables in dataset. When the 
VIF is infinite, it suggests that the information provided by that particular variable is redundant and can be predicted
with perfect accuracy using the other variables in the model.

Perfect multicollinearity can cause issues in regression analysis, as it hinders the estimation of unique contribution 
of each variable to the dependent variable. It can lead to unstable coefficient estimates and inflated standard errors.
"""

# We will remove "Mfg_Month" and "Mfg_Year" and recheck VIF.
X = df2.drop(columns = ['Price', 'Mfg_Month', 'Mfg_Year'], axis = 1)
y = df2['Price']

print('-'*59)
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF for {X.columns[i]} : {vif}")


# Remove "Radio" and recheck VIF.
X = df2.drop(columns = ['Price', 'Radio', 'Mfg_Month', 'Mfg_Year'], axis = 1)
y = df2['Price']

print('-'*59)
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF for {X.columns[i]} : {vif}")

# Remove "Cylinders" and recheck VIF.
X = df2.drop(columns = ['Price', 'Cylinders', 'Radio', 'Mfg_Month', 'Mfg_Year'], axis = 1)
y = df2['Price']

print('-'*59)
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF for {X.columns[i]} : {vif}")

# After removing "Cylinders", other Variables VIF increases,
# Therefore we will keep "Cylinders" even when it's VIF is above 5.
X = df2.drop(columns = ['Price', 'Radio', 'Mfg_Month', 'Mfg_Year'], axis = 1)
y = df2['Price']

print('-'*59)
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF for {X.columns[i]} : {vif}")

# Split the data into Training and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_test.shape: {y_test.shape}")

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, VotingRegressor

# Linear Regression
def linear_regression():
    lm_model = LinearRegression()
    lm_model.fit(X_train, y_train)
    y_lm_pred = lm_model.predict(X_test)
    print(f"MSE: {mean_squared_error(y_test, y_lm_pred)}")
    R2 = r2_score(y_test, y_lm_pred)
    print(f"R-squared: {R2}")


# linear_regression()

# Decision Tree Regressor
def decision_tree_regressor(hyperparam: bool):
    if hyperparam:
        model = DecisionTreeRegressor(
            criterion='squared_error',
            splitter='best',
            min_samples_split=2,
            max_depth=20,
            random_state=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"MSE (hyperparam: True): {mean_squared_error(y_test, y_pred)}")
        print(f"R-squared (hyperparam: True): {r2_score(y_test, y_pred)}")
    else:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"MSE (hyperparam: False): {mean_squared_error(y_test, y_pred)}")
        R2 = r2_score(y_test, y_pred)
        print(f"R-squared (hyperparam: False): {R2}")
"""
criterion = 'squared_error' parameter specifies the function used to measure the quality of a split. For a decision tree
             regressor, the commonly used criterion is mean squared error ('squared_error'), which minimizes the
             variance of the target variable within each leaf.
splitter = 'best' parameter specifies the strategy used to choose the split at each node. 'best' means that the 
            algorithm will choose the best split based on the selected criterion.
min_samples_split = 2 parameter sets the minimum number of samples required to split an internal node. In this case, 
                    a node will only be split if it has at least 2 samples.
max_depth = 20 parameter limits the maximum depth of the decision tree. In this case, the tree will have a maximum depth
            of 20 levels.
random_state = 0 parameter sets the seed for the random number generator. This ensures reproducibility, as using the
              same random seed will result in the same tree structure if the data and other parameters are kept constant
"""
# decision_tree_regressor(hyperparam=False)
# decision_tree_regressor(hyperparam=True)


# RandomForest Regressor
def random_forest_regressor():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"MSE (RandomForest Regressor): {mean_squared_error(y_test, y_pred)}")
    R2 = r2_score(y_test, y_pred)
    print(f"R-squared (RandomForest Regressor): {R2}")


random_forest_regressor()


# Hist Gradient Boosting Regressor

def hist_gradient_boosting_regressor(hyperparam):
    if hyperparam:
        model = HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_depth=3,
            max_iter=130,
            random_state=42,
            scoring='neg_mean_squared_error',
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE (Hist Gradient Boosting Regressor hyperparam: True): {mse}")
        print(f"R-squared (Hist Gradient Boosting Regressor hyperparam: True): {r2}")
    else:
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE (Hist Gradient Boosting Regressor hyperparam: False): {mse}")
        print(f"R-squared (Hist Gradient Boosting Regressor hyperparam: False): {r2}")

"""
learning_rate = 0.1 parameter controls the contribution of each tree to the final prediction. It scales the contribution
                of each tree, and a lower value makes the model more robust but may require more trees.
max_depth = 3 parameter sets the maximum depth of the individual trees in the ensemble. It limits the complexity of 
            each tree, helping to prevent overfitting.
max_iter = 130 parameter specifies the maximum number of boosting iterations (trees) to be fit. This determines the 
            total number of trees in the ensemble.
random_state = 42 parameter sets the seed for the random number generator. This ensures reproducibility, as using the 
               same random seed will result in the same model if the data and other parameters are kept constant.
scoring = 'neg_mean_squared_error' parameter determines the loss function used during training. In this case, it is 
           set to 'neg_mean_squared_error', meaning the negative mean squared error. The negative is used because 
           scikit-learn conventionally maximizes scores, and we want to minimize the mean squared error.
validation_fraction = 0.1 parameter controls the proportion of training data to set aside as a validation set for
                      early stopping. Early stopping can prevent overfitting by monitoring the performance on 
                      the validation set and stopping training when the performance plateaus.
"""
# hist_gradient_boosting_regressor(hyperparam=False)
# hist_gradient_boosting_regressor(hyperparam=True)

# Gradient Boosting Regressor
def gradient_boosting_regressor():
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE (Gradient Boosting Regressor): {mse}")
    print(f"R-squared (Gradient Boosting Regressor): {r2}")

# gradient_boosting_regressor()

# Voting Regressor
def voting_regressor(params: bool):
    if params:
        reg1 = LinearRegression()
        reg2 = DecisionTreeRegressor(criterion='squared_error', splitter='best', min_samples_split=2, max_depth=20,
                                     random_state=0)
        reg3 = RandomForestRegressor()
        reg4 = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=3, max_iter=130, random_state=42,
                                             scoring='neg_mean_squared_error', validation_fraction=0.1)
        reg5 = GradientBoostingRegressor()

        model = VotingRegressor(estimators=[('lr', reg1), ('dt', reg2), ('rf', reg3), ('hgb', reg4), ('gb', reg5)],
                                weights=[1, 1, 1, 2, 2])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE (Voting Regressor params True): {mse}")
        print(f"R-squared (Voting Regressor params True): {r2}")
    else:
        reg1 = LinearRegression()
        reg2 = DecisionTreeRegressor()
        reg3 = RandomForestRegressor()
        reg4 = HistGradientBoostingRegressor()
        reg5 = GradientBoostingRegressor()

        model = VotingRegressor(estimators=[('lr', reg1), ('dt', reg2), ('rf', reg3), ('hgb', reg4), ('gb', reg5)],
                                weights=[1, 1, 1, 2, 2])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE (Voting Regressor params False): {mse}")
        print(f"R-squared (Voting Regressor params False): {r2}")


# voting_regressor(params=True)
# voting_regressor(params=False)

# After review all variations, the best model result was using Gradient Boosting Regressor
# FINAL MODEL
def gradient_boosting_regressor_selected():
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE (Gradient Boosting Regressor): {mse}")
    print(f"R-squared (Gradient Boosting Regressor): {r2}")

    # Add Predicted Values in column 'Y_hat'
    y_hat = model.predict(X)
    df2['Y_hat'] = y_hat

    # Add residual values in column 'Residuals'
    df2['Residuals'] = df2['Price'] - df2['Y_hat']
    print(df2.head())


gradient_boosting_regressor_selected()


# Check Assumptions of Regression
# Check Linearity Assumption
"""
To evaluate linearity, you plot the residuals on the vertical axis against the corresponding Xi values of the 
independent variable on the horizontal axis. If the model is appropriate for the data, you will not see any apparent 
pattern in the residual plot. However, if the model is not appropriate, in the residual plot, there will be a 
relationship between the Xi values and the residuals.
"""

plt.figure(figsize = (20,5))
plt.subplots_adjust(hspace=0.6)

plt.subplot(1,3,1)
sns.scatterplot(df2, x = 'Age_08_04', y = 'Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Age of Car')
plt.ylabel('Residuals')
plt.title('Residuals vs Age of Car')

plt.subplot(1,3,2)
sns.scatterplot(df2, x = 'KM', y = 'Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Kilometer')
plt.ylabel('Residuals')
plt.title('Residuals vs Kilometer')

plt.subplot(1,3,3)
sns.scatterplot(df2, x = 'Weight', y = 'Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Weight')
plt.ylabel('Residuals')
plt.title('Residuals vs Weight')

plt.show() # We can conclude that Linearity Assumption is satisfying.


# Check Independence or Errors Assumption
# To check Independence or errors, we use Durbin-Watson statistic
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(df2['Residuals'].values)) # 2.006181128523849
# Durbin-Watson statistic will approach 0 if successive residuals are positively autocorrelated.#
# If the residuals are not correlated, the value of D will be close to 2.#
# If the residuals are negatively autocorrelated,D will be greater than 2 and could even approach its maximum value of 4
# Here, we get D value close to 2. Therefore we can say that, Independence or Error assumption is satisfying.

# Check Normality Assumption
#You can evaluate the assumption of normality in the errors by constructing a histogram or Normal Probability plot of residuals.
plt.figure(figsize = (12, 6))
sns.histplot(df2, x = 'Residuals', bins = 80, kde = True)
plt.show()

from scipy.stats import probplot
plt.figure(figsize = (12, 6))

probplot(df2['Residuals'], dist = 'norm', plot = plt)
plt.title("Normal Probability Plot")
plt.xlabel("Theoretical Qunatiles")
plt.ylabel("Ordered Residuals")
plt.show() # From Normal Probability Plot, we can say that Normality assumption is satisfying.

# Check Homoscedasticity Assumption
# By ploting Residuals against Y_hat
plt.figure(figsize = (12, 6))
sns.scatterplot(df2, x = 'Y_hat', y = 'Residuals')
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.show()

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(df2['Residuals'], X)
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 0.008033934974226905

# From above plot and Breusch-Pagan Test Result we can say that, Equal Variance assumption is not satisfying.
# To satisfy this assumption, we will do WLS Regression with different weights.

# 1. Weighted Least Square Regression with weight = 1/var(residuals)
residuals = df2['Residuals']
weights = 1 / residuals.var()
wls_model = sm.WLS(y, sm.add_constant(X), weights = weights).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the Breusch-Pagan test results
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 1.124773588963084e-59

# 2. Weighted Least Square Regression with weight = 1/std(residuals)
residuals = df2['Residuals']
weights = 1 / residuals.std()
wls_model = sm.WLS(y, sm.add_constant(X), weights = weights).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the Breusch-Pagan test results
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 1.12477358896347e-59

# 3. Weighted Least Square Regression with weight = 1/y_hat
fitted_values = df2['Y_hat']
weights = 1 / fitted_values
# Handle cases where predicted values are zero
weights[weights == np.inf] = 1e10
wls_model = sm.WLS(y, sm.add_constant(X), weights = weights).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the Breusch-Pagan test results
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 2.5433049852477167e-64

# 4. Weighted Least Square Regression with Residuals and weight = 1/(y_hat)^2
# Store absolute Residuals in residuals
residuals = np.abs(df2['Residuals'])

# Fit a linear regression model with absolute residuals
model2 = sm.OLS(residuals, sm.add_constant(X)).fit()

# Calculate weights for the WLS model with a small constant added
epsilon = 1e-6
weights = 1 / (np.power(model2.fittedvalues,2) + epsilon)
# Fit the WLS model
wls_model = sm.WLS(y, sm.add_constant(X), weights = weights).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the results of the Breusch-Pagan test
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 7.194223544567987e-53

# 5. Weighted Least Square Regression with (Residuals)^2 and weight = 1/(y_hat)^2
# Calculate absolute residuals
abs_residual = np.power(np.abs(df2['Residuals']), 2)
# Fit a linear regression model with absolute residuals
model2 = sm.OLS(abs_residual, sm.add_constant(X)).fit()
# Calculate weights for the WLS model with a small constant added
epsilon = 1e-6
wt2 = 1 / (np.power(model2.fittedvalues, 2) + epsilon)
# Fit the WLS model
wls_model = sm.WLS(y, sm.add_constant(X), weights=wt2).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the results of the Breusch-Pagan test
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 4.3722928903176776e-57

# !!!!!!!!!
# Even After trying various WLS Regression models, Equal Variance Assumption is not Satisying.

# Save Model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}") # MSE: 795185.136811223
print(f"R-squared: {r2}") # R-squared: 0.9404033784746151

import joblib
joblib.dump(model, 'final_model.pkl')












