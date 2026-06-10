import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Cargar el dataset facial de VERA
df = pd.read_csv('datos_rostros_vera.csv')

# 2. Definir variables continuas (X) y objetivo categórico (y)
X = df[["distancia_ojos", "ancho_nariz", "proporcion_mandibula", "simetria_facial"]]
y = pd.factorize(df["ID_Persona"])[0]  # Codificación numérica de identidades

# 3. División de datos (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Instanciar y entrenar la Máquina de Soporte Vectorial con Kernel RBF
# Ajustando hiperparámetros C y gamma para optimización
modelo_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
modelo_svm.fit(X_train, y_train)

# 5. Evaluación del Modelo
y_pred = modelo_svm.predict(X_test)
print(f"Precisión Global (Accuracy Score): {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 6. Desplegar Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - Verificación VERA (SVM)")
plt.show()

# 7. Graficar distribución con Matplotlib
plt.scatter(df["distancia_ojos"], df["simetria_facial"], c=pd.factorize(df["ID_Persona"])[0], cmap="viridis")
plt.xlabel("Distancia entre Ojos")
plt.ylabel("Simetría Facial")
plt.title("Fronteras de Decisión y Distribución Facial (VERA)")
plt.colorbar(label="Índice de Usuario")
plt.show()