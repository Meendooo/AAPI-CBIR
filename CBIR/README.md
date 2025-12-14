# CBIR - Content-Based Image Retrieval System

Sistema de recuperación de imágenes basado en contenido para identificar cartas de la baraja francesa. El proyecto implementa y compara 5 técnicas diferentes de extracción de características, desde histogramas de color hasta Deep Learning.

## Características

- **5 Extractores de Características Implementados:**
  1. **HOG (Histogram of Oriented Gradients)**: Mejor rendimiento (92% accuracy).
  2. **CNN (ResNet50)**: Deep Learning pre-entrenado (90% accuracy).
  3. **Color Histogram**: Histograma en espacio HSV (82% accuracy).
  4. **Local Features (ORB + BoVW)**: Bag of Visual Words (80% accuracy).
  5. **Texture LBP**: Local Binary Patterns (76% accuracy).

- **Búsqueda Eficiente**: Utiliza índices **FAISS** para recuperación ultrarrápida de imágenes similares.
- **Interfaz Gráfica**: Aplicación web interactiva construida con **Streamlit**.
- **Dataset**: 500 imágenes de entrenamiento y 50 de test, divididas en 5 clases (Jotas, Reyes y Joker).

## Resultados de Rendimiento

Comparativa final de los métodos implementados (Top-5):

| Extractor | Accuracy | Precision@5 | Recall@5 | mAP | Interpretación |
|-----------|----------|-------------|----------|-----|----------------|
| **CNN (ResNet50)** | 90.00% | 46.80% | 2.34% | **70.87%** | **Mejor Ranking**. Aunque falla un poco más que HOG, ordena mejor los resultados correctos. |
| **HOG** | **92.00%** | **48.00%** | **2.40%** | 69.05% | **Más Fiable**. Encuentra al menos una carta correcta el 92% de las veces. |
| **Local Features** | 80.00% | 45.60% | 2.28% | 61.25% | **Buen Balance**. Mantiene buena precisión aunque falle en accuracy global. |
| **Color Histogram** | 82.00% | 28.80% | 1.44% | 48.27% | **Engañoso**. Alto accuracy pero baja precisión (muchos falsos positivos del mismo color). |
| **Texture LBP** | 76.00% | 30.40% | 1.52% | 45.79% | **Menos Efectivo**. La textura no discrimina bien entre cartas de baraja. |

### Interpretación de Métricas:
- **Accuracy**: Probabilidad de encontrar *al menos una* carta correcta.
- **Precision@5**: Porcentaje de cartas correctas mostradas (ej. 48% significa ~2.5 cartas correctas de 5).
- **Recall@5**: Porcentaje del total de cartas de esa clase encontradas (bajo por diseño, ya que K=5 << 100).
- **mAP**: Calidad del ordenamiento. Un valor alto indica que los aciertos aparecen en las primeras posiciones.

## Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd CBIR
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Ejecutar la Interfaz Gráfica (Recomendado)

La forma más fácil de usar el sistema es a través de la interfaz web:

```bash
streamlit run app.py
```

La interfaz permite:
- Seleccionar cualquiera de los 5 extractores.
- Subir una imagen propia o usar una del dataset.
- Recortar la imagen (crop) antes de buscar.
- Ver las 11 imágenes más similares del dataset.
- Visualizar métricas de rendimiento en tiempo real.

### 2. Uso desde Línea de Comandos

También puedes realizar consultas individuales desde la terminal:

```bash
# Consultar una imagen específica usando HOG (por defecto)
py query_system.py --image "test/joker/001.jpg" --k 5

# Usar otro extractor (ej: cnn_features)
py query_system.py --image "test/joker/001.jpg" --extractor cnn_features
```

### 3. Reconstruir Índices (Opcional)

Si añades nuevas imágenes al dataset `train/`, puedes reconstruir los índices:

```bash
py build_indexes.py --extractor hog_features
# O para todos:
# py build_indexes.py --extractor color_histogram
# py build_indexes.py --extractor cnn_features
# ...
```

## Estructura del Proyecto

```
CBIR/
├── app.py                  # Interfaz gráfica (Streamlit)
├── build_indexes.py        # Script para crear índices FAISS
├── query_system.py         # Lógica de búsqueda y evaluación
├── requirements.txt        # Dependencias del proyecto
├── extractors/             # Módulos de extracción de características
│   ├── color_histogram.py
│   ├── texture_lbp.py
│   ├── cnn_features.py
│   ├── hog_features.py
│   └── local_features.py
├── faiss_indexes/          # Índices vectoriales pre-calculados
├── mappings/               # Mapeos de índices a nombres de archivo
├── utils/                  # Utilidades de carga de datos
├── train/                  # Imágenes de entrenamiento (Base de datos)
└── test/                   # Imágenes de prueba
```


