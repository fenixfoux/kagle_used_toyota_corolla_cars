"""
INSPIRED BY AN EXAMPLE FROM KAGGLE.COM https://www.kaggle.com/code/vishakhdapat/used-toyota-corolla-cars-price-prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_VISUALISATION = True  # hide/show execution of code section with data visualisation

df = pd.read_csv('ToyotaCorolla.csv')
data = df.copy()
print(data.head())
print(data.info())

print(data.columns)

# - - - - - - - data visualisation : start :   - - - - - - - #
# if DATA_VISUALISATION:
#     fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
#     # show a price histogram
#     # Purpose: Allows to estimate the distribution of car prices in dataset. The KDE curve (Kernel Density Estimate)
#     # provides an approximation of the probability density of the price distribution.
#     # plt.figure(figsize=(10,5))
#     # sns.histplot(data['Price'], bins=30, kde=True)
#     sns.histplot(data['Price'], bins=30, kde=True, ax=axes[0])
#     axes[0].set_title('Price distibution')
#     axes[0].set_xlabel('price')
#     axes[0].set_ylabel('periodicity') # The frequency (number of cars) with a certain price range.
#     # plt.show()
#
#     # The scatter plot for the price depends on the age of the machine
#     # Purpose: Allows to visually assess how the price of a car depends on its age. If there are clear trends (for
#     # example, a decrease in price with increasing age), then they will be visible on the chart.
#     # plt.figure(figsize=(10,5))
#     # sns.scatterplot(x='Age_08_04',y='Price', data=data)
#     sns.scatterplot(x='Age_08_04', y='Price', data=data, ax=axes[1])
#     axes[1].set_title('the dependence of the price on the age of the car')
#     axes[1].set_xlabel('the age of the car')
#     axes[1].set_label('price')
#     # plt.show()
#
#     # boxplot for the price depends on the type of fuel
#     # Purpose: Allows you to compare the price distribution for different types of fuel. The box plot shows the median,
#     # quartiles and emissions, which is useful for identifying possible price differences between different fuel types
#     # plt.figure(figsize=(10,5))
#     # sns.boxplot(x='Fuel_Type', y='Price', data=data)
#     # sns.boxplot(x='Fuel_Type', y='Price', data=data, ax=axes[2])
#     axes[2].set_xlabel('fuel type')
#     axes[2].set_ylabel('price')
#     # plt.show()
#
#     # Автоматическое выравнивание подграфиков
#     # plt.tight_layout()
#     # show plots
#     plt.show()


if DATA_VISUALISATION:
    # show a price histogram
    # Purpose: Allows to estimate the distribution of car prices in dataset. The KDE curve (Kernel Density Estimate)
    # provides an approximation of the probability density of the price distribution.
    plt.figure(figsize=(10,5))
    sns.histplot(data['Price'], bins=30, kde=True)
    plt.title('Price distibution')
    plt.xlabel('price')
    plt.ylabel('periodicity')
    plt.show()

    # The scatter plot for the price depends on the age of the machine
    # Purpose: Allows to visually assess how the price of a car depends on its age. If there are clear trends (for
    # example, a decrease in price with increasing age), then they will be visible on the chart.
    plt.figure(figsize=(10,5))
    sns.scatterplot(x='Age_08_04',y='Price', data=data)
    plt.title('the dependence of the price on the age of the car')
    plt.xlabel('the age of the car')
    plt.ylabel('price')
    plt.show()

    # boxplot for the price depends on the type of fuel
    # Purpose: Allows you to compare the price distribution for different types of fuel. The box plot shows the median,
    # quartiles and emissions, which is useful for identifying possible price differences between different fuel types
    plt.figure(figsize=(10,5))
    sns.boxplot(x='Fuel_Type', y='Price', data=data)
    plt.xlabel('fuel type')
    plt.ylabel('price')
    plt.show()



















# - - - - - - - data visualisation : end :  - - - - - - - #




















