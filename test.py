"""
INSPIRED BY AN EXAMPLE FROM KAGGLE.COM https://www.kaggle.com/code/vishakhdapat/used-toyota-corolla-cars-price-prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

DATA_VISUALISATION = True  # hide/show execution of code section with data visualisation

df = pd.read_csv('ToyotaCorolla.csv')
data = df.copy()
print(data.head())
print(data.info())

print(data.columns)

if DATA_VISUALISATION:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))

    sns.histplot(data['Price'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Price distribution')
    axes[0].set_xlabel('price')
    axes[0].set_ylabel('periodicity')

    sns.scatterplot(x='Age_08_04', y='Price', data=data, ax=axes[1])
    axes[1].set_title('the dependence of the price on the age of the car')
    axes[1].set_xlabel('the age of the car')
    axes[1].set_label('price')

    sns.boxplot(x='Fuel_Type', y='Price', data=data, ax=axes[2])
    axes[2].set_xlabel('fuel type')
    axes[2].set_ylabel('price')

    # plt.tight_layout()

# Функции для переключения между графиками
current_plot = 0


def switch_to_plot1(event):
    global current_plot
    current_plot = 0
    update_plots()


def switch_to_plot2(event):
    global current_plot
    current_plot = 1
    update_plots()


def switch_to_plot3(event):
    global current_plot
    current_plot = 2
    update_plots()


# Функция для обновления отображаемого графика
def update_plots():
    for i, ax in enumerate(axes):
        ax.clear()
        if i == current_plot:
            if i == 0:
                sns.histplot(data['Price'], bins=30, kde=True, ax=ax)
                ax.set_title('Price distribution')
                ax.set_xlabel('price')
                ax.set_ylabel('periodicity')
            elif i == 1:
                sns.scatterplot(x='Age_08_04', y='Price', data=data, ax=ax)
                ax.set_title('the dependence of the price on the age of the car')
                ax.set_xlabel('the age of the car')
                ax.set_ylabel('price')
            elif i == 2:
                sns.boxplot(x='Fuel_Type', y='Price', data=data, ax=ax)
                ax.set_xlabel('fuel type')
                ax.set_ylabel('price')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.draw()


# Добавление кнопок для переключения между графиками
ax_switch1 = plt.axes([0.81, 0.06, 0.1, 0.05])
btn_switch1 = Button(ax_switch1, 'График 1', color='lightgoldenrodyellow', hovercolor='0.975')
btn_switch1.on_clicked(switch_to_plot1)

ax_switch2 = plt.axes([0.81, 0.01, 0.1, 0.05])
btn_switch2 = Button(ax_switch2, 'График 2', color='lightgoldenrodyellow', hovercolor='0.975')
btn_switch2.on_clicked(switch_to_plot2)

ax_switch3 = plt.axes([0.92, 0.06, 0.1, 0.05])
btn_switch3 = Button(ax_switch3, 'График 3', color='lightgoldenrodyellow', hovercolor='0.975')
btn_switch3.on_clicked(switch_to_plot3)

# Отображение графиков
plt.show()
