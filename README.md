# Clasificación de patologías en tomate

Comparación de una MLP y una CNN con un pipeline reproducible de aprendizaje profundo.

## Datos

PlantVillage, 10 clases de tomate y 16.011 imágenes. Split de 12.808 para entrenamiento, 1.601 para validación y 1.602 para test, con semilla 42.

## Resultados

| Modelo | Accuracy | F1 macro | Parámetros |
|---|---:|---:|---:|
| CNN | 94,4% | 93,0% | 4.290.378 |
| MLP | 69,2% | 65,1% | 12.626.762 |

La CNN es el modelo recomendado. La brecha de 5,4 y 15,7 puntos corresponde a la mejor época, que es la que se guarda.

## Pipeline

Carga, EDA, preprocesamiento, aumento, MLP y CNN, entrenamiento, evaluación, auditoría y XAI. Ablación, screening y Random Search. El test se reserva para la evaluación final.

## Estructura

- `data/`: `split.csv` y `class_names.json`.
- `notebooks/`: notebook EP3.
- `models/`: métricas, experimentos e historiales. Los modelos entrenados están en el release `v2.0`.
- `images/`: figuras del notebook y del informe.

## Limitaciones

Imágenes de laboratorio, desbalance de clases, sin validación de campo, uso académico y no diagnóstico autónomo.
