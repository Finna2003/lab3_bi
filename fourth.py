import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Завантаження даних
data = pd.read_csv('futuristic_city_traffic.csv', dtype={'organizer_code': float}).head(7000)

# Вибір ознак та цільової змінної
X = data[[ 'Hour Of Day', 'Speed', 'Random Event Occurred', 'Traffic Density']]
y = data['Is Peak Hour']

# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення класифікатора Random Forest
rf = RandomForestClassifier(random_state=42)

# Сітка параметрів для налаштування
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Пошук найкращих параметрів з використанням GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Отримання найкращих параметрів
best_params = grid_search.best_params_

# Створення моделі Random Forest з найкращими параметрами
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Прогнозування на тестових даних
predictions = best_rf.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
print("Evaluation Results:")
print("Parameters:", best_params)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Виведення класифікаційного звіту
print(classification_report(y_test, predictions))

feature_importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(sorted_feature_importance_df)
