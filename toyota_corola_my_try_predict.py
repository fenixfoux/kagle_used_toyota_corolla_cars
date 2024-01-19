"""
INSPIRED BY AN EXAMPLE FROM KAGGLE.COM https://www.kaggle.com/code/vishakhdapat/used-toyota-corolla-cars-price-prediction
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tkinter import Toplevel
from image_slider import ImageSlider
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_VISUALISATION = False  # hide/show execution of code section with data visualisation
COMBINE_COMFORT_FEATURES = True

all_image_plots_folder = 'all_image_plots'
df = pd.read_csv('ToyotaCorolla.csv')

results_columns = ['model_name', 'combine_comfort_features', 'MSE', 'R-squared', 'hyperparams_used']
model_results = pd.DataFrame(columns=results_columns)
model_results = model_results.astype({
    'model_name': str,
    'combine_comfort_features': bool,
    'MSE': float,
    'R-squared': float,
    'hyperparams_used': bool
})


def check_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)


def add_result(model_name: str, combine_comfort_features: bool, mse: float, r_squared: float,
               hyperparams_used: bool = None):
    global model_results
    new_row = pd.Series({
        'model_name': model_name,
        'combine_comfort_features': combine_comfort_features,
        'MSE': mse,
        'R-squared': r_squared,
        'hyperparams_used': hyperparams_used
    })

    model_results = model_results._append(new_row, ignore_index=True)


data = df.copy()
print(data.head())
print(data.info())
# print(f"ff: {type(data.info())}")
print(data.columns)
# print(data.describe())

# - - - - - - - data visualisation : start :   - - - - - - - #
if DATA_VISUALISATION:
    # show a price histogram
    # Purpose: Allows to estimate the distribution of car prices in dataset. The KDE curve (Kernel Density Estimate)
    # provides an approximation of the probability density of the price distribution.
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Price'], bins=30, kde=True)
    plt.title('Price distibution')
    plt.xlabel('price')
    plt.ylabel('periodicity')
    plt.savefig(f"{all_image_plots_folder}/histogram_price.jpg")

    # The scatter plot for the price depends on the age of the machine
    # Purpose: Allows to visually assess how the price of a car depends on its age. If there are clear trends (for
    # example, a decrease in price with increasing age), then they will be visible on the chart.
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='Age_08_04', y='Price', data=data)
    plt.title('the dependence of the price on the age of the car')
    plt.xlabel('the age of the car')
    plt.ylabel('price')
    plt.savefig(f"{all_image_plots_folder}/scatter_age_price.jpg")

    # boxplot for the price depends on the type of fuel
    # Purpose: Allows you to compare the price distribution for different types of fuel. The box plot shows the median,
    # quartiles and emissions, which is useful for identifying possible price differences between different fuel types
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Fuel_Type', y='Price', data=data)
    plt.xlabel('fuel type')
    plt.ylabel('price')
    plt.savefig(f"{all_image_plots_folder}/boxplot_fuel_price.jpg")

    # create new column 'comfort_level' as summ of comfort features columns
    # Purpose: Boxplot helps to visualize the distribution of prices for different comfort levels.
    comfort_features = ['Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
                        'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
                        'Radio_cassette', 'Parking_Assistant']

    data['Comfort_level'] = data[comfort_features].sum(axis=1)

    # Plotting the price-to-comfort ratio
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='comfort_level', y='Price', data=data)
    plt.title('Соотношение цен к уровню комфорта')
    plt.xlabel('Уровень комфорта')
    plt.ylabel('Цена')
    plt.xticks(rotation=45)  # Поворот подписей по оси X для лучшей читаемости
    plt.savefig(f"{all_image_plots_folder}/boxplot_comfort_price.jpg")

    # show a slider with all created plots
    root = Toplevel()
    root.title("Image Slider")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    image_folder_filepath = "all_image_plots"
    slider = ImageSlider(root, image_folder_filepath)
    root.mainloop()

# # - - - - - - - data visualisation : end :  - - - - - - - #


# - - - - - - - data preparation : start :   - - - - - - - #

# there are columns with datatype: object, transform into numeric and remove those
object_columns = data.select_dtypes(include=['object']).columns.tolist()
print("=" * 99, "\n", f"before transformations columns with datatype: object: {object_columns}")

print(f"all fuel types: {data['Fuel_Type'].unique()}")
data['Fuel_Type_code'] = pd.factorize(data['Fuel_Type'])[0] + 1
print(f"all fuel types as code: {data['Fuel_Type_code'].unique()}")

print(f"al colors: {data['Color'].unique()}")
data['Color_code'] = pd.factorize(data['Color'])[0] + 1
print(f"all colors as code: {data['Color_code'].unique()}")

# after transforming drop initial columns
# also drop model because all cars are the same model (Toyota Corolla)
data.drop(columns=['Model', 'Fuel_Type', 'Color'], axis=1, inplace=True)

object_columns = data.select_dtypes(include=['object']).columns.tolist()
print("=" * 99, "\n", f"after transformations: columns with datatype: object: {object_columns}")

# - - - - - - - data preparation : end :   - - - - - - -  -#

# - - - - - - - checking VIF : start :   - - - - - - - #

# Store Independent variables in X and Dependent Variable in y
X = data.drop(columns=['Price'], axis=1)
y = data['Price']

# Check Multicollinearity Assumption
# correlation_matrix = data.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()

check_vif(X)
# after short look to results: the variance inflation factor (VIF) for the independent
# variables (Age_08_04, Mfg_Month, Mfg_Year) in dataset is infinite (inf), then drop (Mfg_Month, Mfg_Year)
data.drop(columns=['Mfg_Month', 'Mfg_Year'], axis=1, inplace=True)

# recheck VIF
check_vif(X)
# after looking at the result of the VIN check, note that there are several columns with a high VID, and also combine
# all the columns responsible for comfort into one named 'comfort_features' and recheck the VIF
if COMBINE_COMFORT_FEATURES:
    comfort_features = ['Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
                        'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
                        'Radio_cassette', 'Parking_Assistant']
    data['Comfort_level'] = data[comfort_features].sum(axis=1)

    #  redefine X after added new column
    X = data.drop(columns=['Price'], axis=1)

    # recheck VIF expecting to get (infinite) as result of the VIF for all combined columns
    check_vif(X)
    # the result is as was expected, now remove all those column and redefine X and recheck the VIF
    print(f"len data columns before removing comfort features: {len(data.columns)}")
    data.drop(columns=comfort_features, axis=1, inplace=True)
    print(f"len data columns after removing comfort features: {len(data.columns)}")
    X = data.drop(columns=['Price'], axis=1)
    check_vif(X)
    print(data.columns)

# currently observe increased VIF for 'Cylinders', try to remove that column and do all necessary operation
# print(data.columns)
X = data.drop(columns=['Price', 'Cylinders'], axis=1)
check_vif(X)

# removing 'Cylinders' increase multiple another columns VIF, so decide to stop here and returning 'Cylinders'
X = data.drop(columns='Price')
# print(X.columns)
# print(len(X.columns))
check_vif(X)

# - - - - - - - checking VIF : end :   - - - - - - - #

# - - - - - - - testing different approaches of models : start :   - - - - - - - #
# will check next models:
#     Linear Regression
#     Decision Tree Regressor
#     RandomForest Regressor
#     Hist Gradient Boosting Regressor
#     Gradient Boosting Regressor


# split the data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# checking the shapes of training and testing splits
print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_test.shape: {y_test.shape}")

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor


#     Linear Regression
def linear_regression(X_train, y_train, X_test, y_test):
    global COMBINE_COMFORT_FEATURES
    lm_model = LinearRegression()
    lm_model.fit(X_train, y_train)
    y_lm_pred = lm_model.predict(X_test)
    linear_reg_mse = mean_squared_error(y_test, y_lm_pred)
    print(f"MSE: {linear_reg_mse}")
    linear_reg_r2_score = r2_score(y_test, y_lm_pred)
    print(f"R2 score: {linear_reg_r2_score}")
    add_result('linear_regression', COMBINE_COMFORT_FEATURES, linear_reg_mse, linear_reg_r2_score)


linear_regression(X_train, y_train, X_test, y_test)


# result:
#   - with combine comfort features:
#       - MSE: 1700090.7866237424
#       - R2 score: 0.8725835500704757
#   - without combine comfort features:
#       - MSE: 1451212.433999483
#       - R2 score: 0.8912361987438254


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
        tree_reg_mse = mean_squared_error(y_test, y_pred)
        tree_reg_r_squared = r2_score(y_test, y_pred)
        print(f"MSE (hyperparam: True): {tree_reg_mse}")
        print(f"R-squared (hyperparam: True): {tree_reg_r_squared}")
        add_result('decision_tree_regressor', COMBINE_COMFORT_FEATURES, tree_reg_mse, tree_reg_r_squared, True)
    else:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tree_reg_mse = mean_squared_error(y_test, y_pred)
        print(f"MSE (hyperparam: False): {tree_reg_mse}")
        tree_reg_r_squared = r2_score(y_test, y_pred)
        print(f"R-squared (hyperparam: False): {tree_reg_r_squared}")
        add_result('decision_tree_regressor', COMBINE_COMFORT_FEATURES, tree_reg_mse, tree_reg_r_squared, False)


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

decision_tree_regressor(hyperparam=False)
# result:
#   - with combine comfort features:
#       - MSE (hyperparam: False): 1628104.5208333333
#       - R-squared (hyperparam: False): 0.8779786939668275
#   - without combine comfort features:
#       - MSE (hyperparam: False): 1686094.5416666667
#       - R-squared (hyperparam: False): 0.8736325245480777

decision_tree_regressor(hyperparam=True)


# result:
#   - with combine comfort features:
#       - MSE (hyperparam: True): 1894145.0432098764
#       - R-squared (hyperparam: True): 0.8580397947237264
#   - without combine comfort features:
#       - MSE (hyperparam: True): 1692528.3090277778
#       - R-squared (hyperparam: True): 0.8731503339478608


# RandomForest Regressor
def random_forest_regressor():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rn_forest_reg_mse = mean_squared_error(y_test, y_pred)
    print(f"MSE (RandomForest Regressor): {rn_forest_reg_mse}")
    rn_forest_reg_r_squared = r2_score(y_test, y_pred)
    print(f"R-squared (RandomForest Regressor): {rn_forest_reg_r_squared}")
    add_result('random_forest_regressor', COMBINE_COMFORT_FEATURES, rn_forest_reg_mse, rn_forest_reg_r_squared)


random_forest_regressor()


# result:
#   - with combine comfort features:
#       - MSE (RandomForest Regressor): 976000.0723611112
#       - R-squared (RandomForest Regressor): 0.9268518685415744
#   - without combine comfort features:
#       - MSE (RandomForest Regressor): 935806.9698989582
#       - R2-squared (RandomForest Regressor): 0.9298642149807618


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
        add_result('hist_gradient_boosting_regressor', COMBINE_COMFORT_FEATURES, mse, r2, True)
    else:
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE (Hist Gradient Boosting Regressor hyperparam: False): {mse}")
        print(f"R-squared (Hist Gradient Boosting Regressor hyperparam: False): {r2}")
        add_result('hist_gradient_boosting_regressor', COMBINE_COMFORT_FEATURES, mse, r2, False)


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

hist_gradient_boosting_regressor(hyperparam=False)
# result:
#   - with combine comfort features:
#       - MSE (Hist Gradient Boosting Regressor hyperparam: False): 1023050.1435537884
#       - R-squared (Hist Gradient Boosting Regressor hyperparam: False): 0.9233256138924284
#   - without combine comfort features:
#       - MSE (Hist Gradient Boosting Regressor hyperparam: False): 954629.1526114694
#       - R-squared (Hist Gradient Boosting Regressor hyperparam: False): 0.928453551667942

hist_gradient_boosting_regressor(hyperparam=True)


# result:
#   - with combine comfort features:
#       - MSE (Hist Gradient Boosting Regressor hyperparam: True): 940857.4444660763
#       - R-squared (Hist Gradient Boosting Regressor hyperparam: True): 0.9294856978186992
#   - without combine comfort features:
#       - MSE (Hist Gradient Boosting Regressor hyperparam: True): 891152.7170746825
#       - R-squared (Hist Gradient Boosting Regressor hyperparam: True): 0.9332109105889557

# Gradient Boosting Regressor
def gradient_boosting_regressor():
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE (Gradient Boosting Regressor): {mse}")
    print(f"R-squared (Gradient Boosting Regressor): {r2}")
    add_result('gradient_boosting_regressor', COMBINE_COMFORT_FEATURES, mse, r2)
    return model


gradient_boosting_regressor()
# result:
#   - with combine comfort features:
#       - MSE (Gradient Boosting Regressor): 899834.2401939466
#       - R-squared (Gradient Boosting Regressor): 0.9325602577740937
#   - without combine comfort features:
#       - MSE (Gradient Boosting Regressor): 852500.6772306839
#       - R-squared (Gradient Boosting Regressor): 0.9361077592385724

print("==" * 99)
print(f"model_results:\n{model_results.head(50)}")
print(f"COMBINE_COMFORT_FEATURES: {COMBINE_COMFORT_FEATURES}")
# - - - - - - - testing different approaches of models : end :   - - - - - - - #

# - - - - - - - review : start :   - - - - - - - #
# after a short review of results, the best result:
#       MSE (Gradient Boosting Regressor): 902644.5464443122
#       R-squared (Gradient Boosting Regressor): 0.9323496341718404

# Add Predicted Values in column 'Y_predicted'
best_model = gradient_boosting_regressor()
y_predicted = best_model.predict(X)
data['Y_predicted'] = y_predicted
# Add residual values in column 'Residuals'
data['Residuals'] = data['Price'] - data['Y_predicted']
print(data[['Id', 'Price', 'Y_predicted', 'Residuals']].head(15))

#          Check Assumptions of Regression          #
"""
To evaluate linearity, plot the residuals on the vertical axis against the corresponding Xi values of the 
independent variable on the horizontal axis. If the model is appropriate for the data, you will not see any apparent 
pattern in the residual plot. However, if the model is not appropriate, in the residual plot, there will be a 
relationship between the Xi values and the residuals.
"""
plt.figure(figsize=(15, 5))
plt.subplots_adjust(hspace=0.6)

plt.subplot(1, 3, 1)
sns.scatterplot(data, x='Age_08_04', y='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Age of Car')
plt.ylabel('Residuals')
plt.title('Residuals vs Age of Car')

plt.subplot(1, 3, 2)
sns.scatterplot(data, x='KM', y='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Kilometer')
plt.ylabel('Residuals')
plt.title('Residuals vs Kilometer')

plt.subplot(1, 3, 3)
sns.scatterplot(data, x='Weight', y='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Weight')
plt.ylabel('Residuals')
plt.title('Residuals vs Weight')

plt.show()
# In general, based on these graphs, the following conclusions can be drawn:
#
# - The assumption of linear regression is fulfilled for the dependence of the price of a car on its age,
#   but it is not ideal.
# - The assumption of linear regression is not fulfilled for the dependence of the price of a car on its mileage.
# - The assumption of linear regression can be made for the dependence of the price of a car on its weight, but a
#   more thorough analysis of the data is necessary.

# Statistical tests should be used to more accurately analyze the assumption of linear regression. One such test
# is the Durbin-Watson test. This test allows to assess how well the data corresponds to a linear relationship.
# If the Durbin-Watson test shows that the data does not correspond to a linear relationship, then it is necessary
# to use a regression model that takes into account the nonlinear dependence of the data.

#               Durbin-Watson test              #
# Durbin-Watson statistic will :
#           - approach 0 if successive residuals are positively autocorrelated
#           - close to 2 if the residuals are not correlated
#           - greater than 2 and could even approach its maximum value of 4 residuals are negatively auto correlated
#

from statsmodels.stats.stattools import durbin_watson

durbin_watson_test_results = durbin_watson(data['Residuals'].values)
print(durbin_watson_test_results)
# Durbin test result value is close to 2. Therefore, we can say that, Independence or Error assumption is satisfying.

#               Check Normality Assumption              #
# evaluate the assumption of normality in the errors by constructing a histogram or Normal Probability plot of residuals
plt.figure(figsize=(12, 6))
sns.histplot(data, x='Residuals', bins=80, kde=True)
plt.show()

from scipy.stats import probplot

plt.figure(figsize=(12, 6))

probplot(data['Residuals'], dist='norm', plot=plt)
plt.title("Normal Probability Plot")
plt.xlabel("Theoretical Qunatiles")
plt.ylabel("Ordered Residuals")
plt.show()
# Normal Probability Plot tell us that Normality assumption is satisfying.

# Check Homoscedasticity Assumption by plotting Residuals against Y_predicted
plt.figure(figsize=(12, 6))
sns.scatterplot(data, x='Y_predicted', y='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

#                Breusch-Pagan Test                 #
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(data['Residuals'], X)
print("Breusch-Pagan test p-value:", bp_test[1])


# From above plot and Breusch-Pagan Test Result we can say that, Equal Variance assumption is not satisfying.
# To satisfy this assumption, we will do WLS Regression with different weights.

# 1. Weighted Least Square Regression with weight = 1/var(residuals)
residuals = data['Residuals']
weights = 1 / residuals.var()
wls_model = sm.WLS(y, sm.add_constant(X), weights = weights).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the Breusch-Pagan test results
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 1.124773588963084e-59

# 2. Weighted Least Square Regression with weight = 1/std(residuals)
residuals = data['Residuals']
weights = 1 / residuals.std()
wls_model = sm.WLS(y, sm.add_constant(X), weights = weights).fit()
# Perform the Breusch-Pagan test for homoscedasticity
bp_test = het_breuschpagan(wls_model.resid, wls_model.model.exog)
# Display the Breusch-Pagan test results
print("Breusch-Pagan test p-value:", bp_test[1]) # Breusch-Pagan test p-value: 1.12477358896347e-59

# 3. Weighted Least Square Regression with weight = 1/y_hat
fitted_values = data['Y_predicted']
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
residuals = np.abs(data['Residuals'])

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
abs_residual = np.power(np.abs(data['Residuals']), 2)
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

import joblib
best_model_name = best_model.__class__.__name__ + 'model.pkl'
print(best_model_name)
joblib.dump(best_model, best_model_name)

# - - - - - - - review : end :   - - - - - - - #
