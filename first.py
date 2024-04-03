import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Завантаження даних
data = pd.read_csv("futuristic_city_traffic.csv").head(7000)

# Вибір ознак та цільової змінної
X = data[[ 'Hour Of Day', 'Speed', 'Is Peak Hour', 'Traffic Density']]
y = data[ 'Random Event Occurred']

# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення моделі логістичної регресії
log_reg = LogisticRegression()

# Тренування моделі
log_reg.fit(X_train, y_train)

# Прогнозування на тестових даних
predictions = log_reg.predict(X_test)

# Оцінка моделі
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

# Виведення метрик
print("Accuracy:", accuracy)
print("F1 Score:", f1)


