# 🧠 Clasificación de Riesgo Crediticio – German Credit Data

## 📋 Descripción General
Este proyecto desarrolla un modelo de **clasificación supervisada** para predecir el riesgo crediticio de clientes utilizando el dataset **Statlog (German Credit Data)** del repositorio UCI.  
El objetivo principal es **identificar si un solicitante de crédito representa un “buen” o “mal” riesgo crediticio**, a partir de variables demográficas y financieras.

---

## 📦 Contenido del Proyecto
- `EntregableM7.ipynb` → Cuaderno principal con todo el flujo de análisis.
- `README.md` → Documento explicativo del proyecto.

---

## ⚙️ Flujo del Proyecto

### 1. Importación y Carga de Datos
Se utiliza la librería `ucimlrepo` para obtener el dataset **German Credit (id=144)** directamente desde el repositorio UCI.

### 2. Análisis Exploratorio (EDA)
- Inspección de tipos de variables y valores faltantes.  
- Análisis de distribución de las clases (“buen crédito” vs “mal crédito”).  
- Identificación de variables categóricas y numéricas.

### 3. Preprocesamiento de Datos
- **Codificación categórica:** Se aplica `OneHotEncoder` de `sklearn` para transformar variables categóricas en numéricas.  
- **Estandarización:** Se normalizan las variables numéricas con `StandardScaler`.  
- **Balanceo de clases:** Se utiliza `SMOTE` para mitigar el desbalance entre clases.

### 4. División del Conjunto de Datos
Se realiza una división `train_test_split` para evaluar el modelo con datos no vistos.

### 5. Modelado con Red Neuronal
Se implementa una red neuronal utilizando `TensorFlow` y `Keras` con la siguiente estructura:
- Varias capas densas (`Dense`) con activación ReLU.  
- Dropout para evitar sobreajuste.  
- EarlyStopping y ReduceLROnPlateau para optimizar el entrenamiento.

### 6. Evaluación del Modelo
Se evalúa el rendimiento utilizando:
- **Matriz de confusión**  
- **Reporte de clasificación (precision, recall, f1-score)**  
- **Curva ROC y AUC**

### 7. Interpretabilidad del Modelo
Se emplea **SHAP (SHapley Additive exPlanations)** para analizar la contribución de cada variable a las predicciones del modelo, ayudando a entender cómo influyen las características en la clasificación del riesgo.

---

## 📊 Resultados Destacados
- Buen desempeño en la predicción del riesgo crediticio tras balancear las clases.  
- SHAP permitió identificar las variables con mayor impacto en las decisiones del modelo.  
- Se demuestra la aplicabilidad de redes neuronales en problemas de clasificación tabular.

---

## 🧩 Tecnologías Utilizadas
- **Python 3.x**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **TensorFlow / Keras**
- **SHAP**
- **ucimlrepo**

---

## 🚀 Ejecución del Proyecto
1. Instalar dependencias:
   ```bash
   pip install pandas numpy scikit-learn tensorflow imbalanced-learn shap ucimlrepo seaborn matplotlib

