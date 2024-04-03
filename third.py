import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Завантаження даних
data = pd.read_csv('futuristic_city_traffic.csv').head(7000)

# Вибір ознак та цільової змінної
X = data[['Random Event Occurred', 'Traffic Density', 'Speed', 'Hour Of Day']]
y = data['Is Peak Hour']
print("1")
# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання моделі методу опорних векторів
svm = SVC()
svm.fit(X_train, y_train)

# Прогнозування на тестових даних
predictions = svm.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

# Виведення метрик
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Виведення класифікаційного звіту
print("\nClassification Report:")
print(classification_report(y_test, predictions))
