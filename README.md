# EP2 - MLP vs CNN para patologías en tomate

**Asignatura:** Técnicas Avanzadas de Machine Learning (TLY1102)  
**Sección:** 001D  
**Integrantes:** Matías Gutiérrez, David Larenas, Victor Mena

Comparamos un MLP y una CNN para clasificar 10 clases de hojas de tomate del dataset PlantVillage. Usamos la misma variante de la EP1, un split estratificado 80/10/10 y el mismo protocolo de entrenamiento para ambos modelos.

## Resultados

- MLP: 71,5% accuracy, 64,6% F1 macro, 25.331.850 parámetros.
- CNN: 94,1% accuracy, 93,2% F1 macro, 3.305.930 parámetros.

La CNN usa menos parámetros y obtiene mejores métricas. El MLP no sobreajusta, pero se queda corto por underfitting.

## Estructura

- `EP2_Gutierrez_Larenas_Mena.ipynb`: notebook ejecutado.
- `data/split.csv`: split estratificado con rutas relativas.
- `data/class_names.json`: orden de las clases.
- `models/`: modelos, historiales, métricas y reportes.
- `images/`: figuras del notebook.

El modelo MLP pesa 290 MB y está en [Releases v1.0](https://github.com/mtgutierrezh/tly1102-ep2-tomate/releases/tag/v1.0). El CNN (`models/cnn_ep2.keras`, 39,7 MB) está en el repo.

## Reproducibilidad

Fijamos `SEED = 42` para Python, NumPy y TensorFlow. El split se guarda en `data/split.csv`, así que no depende del entorno.

## Limitaciones

Las imágenes son de laboratorio, con hoja centrada y fondo uniforme. El dataset está desbalanceado, usamos una sola semilla y no hicimos Data Augmentation ni validación externa. El test es una partición estratificada 80/10/10 del mismo dataset.
