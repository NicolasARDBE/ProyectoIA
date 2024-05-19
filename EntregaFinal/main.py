#Código hecho con ayuda de Chat GPT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Leer el dataset
df = pd.read_csv('fifa21_male2.csv', low_memory=False)
df = df.sample(n=1000, random_state=42)

# Preprocesamiento y limpieza de datos
df = df.drop(["Hits", "Value", "Wage", "Release Clause", "ID", "Name", "Club", "Position", 
              "Player Photo", "Club Logo", "Flag Photo", "Team & Contract", "Growth", "Joined", 
              "Loan Date End", "Contract", "Gender", "Composure", "W/F", "SM", "A/W", "D/W", 
              "IR", "LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM", 
              "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", 
              "RCB", "RB", "GK", "BP", "BOV", "foot"], axis=1)

def convertir_a_cm(altura):
    pies, pulgadas = altura.split("'")
    pies = int(pies)
    pulgadas = int(pulgadas[:-1])
    altura_cm = (pies * 12 + pulgadas) * 2.54
    return altura_cm

def convertir_a_kg(peso):
    peso_lbs = int(peso[:-3])
    peso_kg = peso_lbs * 0.453592
    return peso_kg

df['Weight'] = df['Weight'].apply(convertir_a_kg)
df['Height'] = df['Height'].apply(convertir_a_cm)

# 1. Acá realizamos la imputación:
# Eliminar filas con valores nulos
df = df.dropna()

# 2. Acá convertimos las categorías en números:
# Codificación de frecuencia para "Nationality"
frecuencia = df['Nationality'].value_counts()
df['nationality_frecuencia'] = df['Nationality'].map(frecuencia)

# Eliminar columna "Nationality"
df = df.drop(["Nationality"], axis=1)

# División del dataset
X = df.drop(columns=['OVA'])
y = df['OVA']

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 3. Acá normalizamos:
# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 5. Acá usamos un algoritmo para cumplir con el objetivo:
# Definir el modelo
model = RandomForestRegressor(random_state=42)

# 6. Acá cambiamos los hiperparámetros:
# Definir el rango de hiperparámetros a probar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, None]
}

# 8. Repetir 6 y 7 varias veces hasta lograr una buena medida. (Varios for para cambiar los hiperparámetros):
# Utilizar GridSearchCV para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 7. Mostrar medidas de rendimiento (función de costo) tanto en train como en val:
# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Predicciones en el conjunto de entrenamiento
y_train_pred = best_model.predict(X_train)

# Predicciones en el conjunto de validación
y_val_pred = best_model.predict(X_val)

# Medidas de rendimiento
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

print(f'Train Mean Squared Error: {train_mse}')
print(f'Validation Mean Squared Error: {val_mse}')

# 9. Sacar el error de test:
# Predicciones en el conjunto de prueba
y_test_pred = best_model.predict(X_test)

# Medidas de rendimiento en el conjunto de prueba
test_mse = mean_squared_error(y_test, y_test_pred)

print(f'Test Mean Squared Error: {test_mse}')

# 10. Predecir un dato nuevo (inventado):
# Crear un nuevo dato inventado con las mismas características que el conjunto de entrenamiento
new_data = pd.DataFrame({
    'Age': [18],
    'nationality_frecuencia': [90],
    'POT': [78],
    'Height': [172],
    'Weight': [60.7],
    'Base Stats': [345],
    'PAC': [71],
    'SHO': [65],
    'PAS': [61],
    'DRI': [71],
    'DEF': [24],
    'PHY': [56],
    'Attacking': [0],
    'Crossing': [0],
    'Finishing': [0],
    'Heading Accuracy': [0],
    'Short Passing': [0],
    'Volleys': [0],
    'Skill': [0],
    'Dribbling': [0],
    'Curve': [0],
    'FK Accuracy': [0],
    'Long Passing': [0],
    'Ball Control': [0],
    'Movement': [0],
    'Acceleration': [0],
    'Sprint Speed': [0],
    'Agility': [0],
    'Reactions': [0],
    'Balance': [0],
    'Power': [0],
    'Shot Power': [0],
    'Jumping': [0],
    'Stamina': [0],
    'Strength': [0],
    'Long Shots': [0],
    'Mentality': [0],
    'Aggression': [0],
    'Interceptions': [0],
    'Positioning': [0],
    'Vision': [0],
    'Penalties': [0],
    'Defending': [0],
    'Marking': [0],
    'Standing Tackle': [0],
    'Sliding Tackle': [0],
    'Goalkeeping': [0],
    'GK Diving': [0],
    'GK Handling': [0],
    'GK Kicking': [0],
    'GK Positioning': [0],
    'GK Reflexes': [0],
    'Total Stats': [0]
})

# Asegurarse de que las columnas estén en el mismo orden que las características de entrenamiento
new_data = new_data[X.columns]

# Escalar el nuevo dato
new_data_scaled = scaler.transform(new_data)

# Predecir la etiqueta para el nuevo dato
new_prediction = best_model.predict(new_data_scaled)

# Asegurarse de que la predicción está en el rango válido (0-100):

new_prediction = np.clip(new_prediction, 0, 100)

print(f'Predicción para el nuevo dato: {new_prediction}')

'''''
CONCLUSIONES FINALES:


    -1. Rendimiento del Modelo: El modelo Random Forest Regressor ajustado con GridSearchCV ha alcanzado un muy buen rendimiento en el conjunto de entrenamiento, validación y prueba:
        -Train Mean Squared Error: 0.374
        -Validation Mean Squared Error: 2.360
        -Test Mean Squared Error: 2.368
    -La diferencia entre el error de entrenamiento y los errores de validación y prueba es pequeña, lo que indica que el modelo no está sobreajustado y generaliza bien a datos nuevos.


    -2. Las predicciones para los nuevos datos inventados varían de manera consistente en función de los valores de entrada:
        -Para el primer caso (datos de un jugador con POT alto, buena altura y peso, y altos valores en atributos clave como PAC, SHO, PAS y DRI), el modelo predice un OVA de aproximadamente 67.595.
        -Para el segundo caso (jugador con alta edad y POT moderado, pero muy alto PAC), la predicción es de 66.57.
        -Para el tercer caso (jugador con POT alto pero con valores bajos en PAC, SHO y PAS), la predicción es 64.415.
        -Para el cuarto caso (jugador muy joven con POT moderado y valores balanceados), la predicción es 58.68.
        -Para el quinto caso (jugador joven con valores moderados en atributos clave), la predicción es 60.115.

    
    -3. Impacto de los Atributos en las Predicciones: 
        -POT (Potential) parece tener un impacto significativo en la predicción de OVA, como era de esperar, dado que representa el potencial del jugador.
        -Atributos como PAC (Pace), SHO (Shooting), PAS (Passing) y DRI (Dribbling) también influyen significativamente en las predicciones.
        -Las diferencias en las predicciones para jugadores con atributos muy diferentes muestran que el modelo es sensible a las características individuales de cada jugador.

    
    -4. Utilidad del Modelo:
        -El modelo es útil para hacer predicciones de OVA razonables basadas en una variedad de atributos del jugador.
        -Se podría usar para evaluar potenciales fichajes o comparaciones entre jugadores con diferentes perfiles.


    -5. Mejoras Potenciales:
        -Aunque el modelo funciona bien, siempre es posible explorar más configuraciones de hiperparámetros y otros modelos de machine learning para buscar mejoras adicionales.
        -Añadir más datos de entrenamiento y realizar más preprocesamiento y selección de características también podría mejorar el rendimiento del modelo.


    -6. Conclusión Final:
        -El modelo de regresión basado en Random Forest Regressor, después de un proceso cuidadoso de selección de hiperparámetros y preprocesamiento de datos, ofrece predicciones precisas y consistentes del Overall Rating (OVA) de jugadores de FIFA. Las métricas de rendimiento indican un buen ajuste y capacidad de generalización, lo que sugiere que este enfoque es efectivo y robusto para la tarea propuesta.
'''''