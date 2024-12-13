import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Daten laden
excel_path = '/home/fit_member/Documents/NS_SemesterWork/Project/data/data_overview_binary_cleaned_256.xlsx'
data = pd.read_excel(excel_path)
good_data = data[data['quality'] == 'good']

# Features und Labels auswählen
X = good_data[['Volume_mL', 'Surface_mm2', 'Mean_Intensity', 'Min_Intensity', 'Max_Intensity', 'Std_Intensity', 'Compactness']].values
y = good_data['Classification'].values  # Zielvariable: 0 = healthy, 1 = pathological

# 2. Daten skalieren
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Random Forest Classifier initialisieren
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. K-Fold Cross-Validation einrichten
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-Validation ausführen
print("Starte K-Fold Cross-Validation...")
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Ergebnisse der Cross-Validation ausgeben
print("\n--- Cross-Validation Ergebnisse ---")
print(f"Genauigkeit pro Fold: {scores}")
print(f"Durchschnittliche Genauigkeit: {np.mean(scores):.4f}")
print(f"Standardabweichung der Genauigkeit: {np.std(scores):.4f}")

# 5. Modell auf den gesamten Daten trainieren, um Feature Importance zu berechnen
model.fit(X, y)

# 6. Feature Importance berechnen und anzeigen
importances = model.feature_importances_
feature_names = ['Volume_mL', 'Mean_Intensity', 'Min_Intensity', 'Max_Intensity', 'Surface_mm2', 'Std_Intensity', 'Compactness']

# Sortiere die Features nach Wichtigkeit
sorted_indices = np.argsort(importances)[::-1]

print("\n--- Feature Importance ---")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# 7. Feature Importance visualisieren
plt.figure(figsize=(8, 6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[sorted_indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in sorted_indices], rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# 8. Modell evaluieren (optional)
# Teile die Daten manuell auf, um Predictions und Metriken zu berechnen
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ergebnisse ausgeben
print("\n--- Modell Evaluation auf Test-Daten ---")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Pathological']))
