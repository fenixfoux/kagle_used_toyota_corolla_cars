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


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_VISUALISATION = True  # hide/show execution of code section with data visualisation
all_image_plots_folder = 'all_image_plots'

df = pd.read_csv('ToyotaCorolla.csv')
data = df.copy()
print(data.head())
print(data.info())

print(data.columns)

# - - - - - - - data visualisation : start :   - - - - - - - #


if DATA_VISUALISATION:
    # show a price histogram
    # Purpose: Allows to estimate the distribution of car prices in dataset. The KDE curve (Kernel Density Estimate)
    # provides an approximation of the probability density of the price distribution.
    plt.figure(figsize=(10,5))
    sns.histplot(data['Price'], bins=30, kde=True)
    plt.title('Price distibution')
    plt.xlabel('price')
    plt.ylabel('periodicity')
    plt.savefig(f"{all_image_plots_folder}/histogram_price.jpg")

    # The scatter plot for the price depends on the age of the machine
    # Purpose: Allows to visually assess how the price of a car depends on its age. If there are clear trends (for
    # example, a decrease in price with increasing age), then they will be visible on the chart.
    plt.figure(figsize=(10,5))
    sns.scatterplot(x='Age_08_04',y='Price', data=data)
    plt.title('the dependence of the price on the age of the car')
    plt.xlabel('the age of the car')
    plt.ylabel('price')
    plt.savefig(f"{all_image_plots_folder}/scatter_age_price.jpg")

    # boxplot for the price depends on the type of fuel
    # Purpose: Allows you to compare the price distribution for different types of fuel. The box plot shows the median,
    # quartiles and emissions, which is useful for identifying possible price differences between different fuel types
    plt.figure(figsize=(10,5))
    sns.boxplot(x='Fuel_Type', y='Price', data=data)
    plt.xlabel('fuel type')
    plt.ylabel('price')
    plt.savefig(f"{all_image_plots_folder}/boxplot_fuel_price.jpg")

    # show a slider with all created plots
    root = Toplevel()
    root.title("Image Slider")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    image_folder_filepath = "all_image_plots"
    slider = ImageSlider(root, image_folder_filepath)
    root.mainloop()

# # - - - - - - - data visualisation : end :  - - - - - - - #



















