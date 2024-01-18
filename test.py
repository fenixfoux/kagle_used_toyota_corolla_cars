import pandas as pd

results_columns = ['model_name', 'combine_comfort_features', 'MSE', 'R-squared', 'hyperparams_used']
model_results = pd.DataFrame(columns=results_columns)

# Задаем типы данных для колонок
model_results = model_results.astype({
    'model_name': str,
    'combine_comfort_features': bool,
    'MSE': float,
    'R-squared': float,
    'hyperparams_used': str
})

# Создаем новую строку
new_row = pd.Series({
    'model_name': 'Linear Regression',
    'combine_comfort_features': True,
    'MSE': 0.123,
    'R-squared': 0.876,
    'hyperparams_used': 'alpha=0.01'
})

# Добавляем строку в DataFrame
model_results = model_results._append(new_row, ignore_index=True)
print(model_results.head())
