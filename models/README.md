# Modelos

Los modelos finales se publican como assets del release `v2.0`. No se versionan en git porque la MLP pesa cerca de 97 MB y la CNN cerca de 17 MB.

Enlace del release: https://github.com/mtgutierrezh/tly1102-tomate/releases/tag/v2.0

## Archivos del release

| Archivo | Contenido |
|---|---|
| `cnn_ep3_final.keras` | CNN final con la mejor configuración y la semilla 42 |
| `mlp_ep3_final.keras` | MLP final con la mejor configuración y la semilla 42 |
| `metricas_finales.json` | Métricas de test, configuración y resultados por semilla |
| `experimentos.csv` | Las 30 corridas de ablación, screening, búsqueda y evaluación final |
| `comparacion_final.csv` | Tabla comparativa de la CNN y la MLP |
| `figuras.zip` | Figuras del notebook y del informe |
| `split.csv` | Particiones usadas en todas las corridas |

## Entrada y salida

La entrada es una imagen RGB de 128 x 128 píxeles. La salida es un vector de 10 probabilidades en el orden de `data/class_names.json`. La clase predicha es el índice con mayor probabilidad.

## Carga

```python
import tensorflow as tf

modelo = tf.keras.models.load_model("cnn_ep3_final.keras", compile=False)
probabilidades = modelo.predict(imagen)
```

El modelo se guardó sin optimizador para reducir el tamaño. La arquitectura y los pesos se conservan y sirven para inferencia, Grad-CAM y LIME.

## Artefactos de experimentación

| Ruta | Contenido |
|---|---|
| `historial/` | Curvas de entrenamiento por corrida |
| `reportes/` | Reportes de clasificación y matrices de confusión |
| `experimentos.csv` | Registro de hiperparámetros y métricas de validación |
| `metricas_finales.json` | Resumen final de la CNN y la MLP |

## Limitaciones

Los modelos se entrenaron con imágenes de laboratorio, con hoja centrada y fondo uniforme. El desempeño en fotografías de campo puede ser menor. La clase sana y las clases minoritarias tienen menos ejemplos. El modelo no reemplaza la inspección de un especialista y debe usarse con supervisión humana.
