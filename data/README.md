# Datos

## Origen

Usamos el subconjunto de tomate de PlantVillage, disponible en Kaggle como `emmarex/plantdisease`. El subconjunto tiene 10 clases y 16.011 imágenes de hojas. Las imágenes originales son de 256 x 256 píxeles y el modelo las recibe a 128 x 128.

Referencia del dataset: Hughes y Salathé, 2015, An open access repository of images on plant health to enable the development of mobile disease diagnostics, arXiv:1511.08060.

## Clases

El orden de las clases es el de `class_names.json` y se mantiene en el entrenamiento, la evaluación y las figuras.

| Índice | Clase |
|---|---|
| 0 | Tomato_Bacterial_spot |
| 1 | Tomato_Early_blight |
| 2 | Tomato_Late_blight |
| 3 | Tomato_Leaf_Mold |
| 4 | Tomato_Septoria_leaf_spot |
| 5 | Tomato_Spider_mites_Two_spotted_spider_mite |
| 6 | Tomato__Target_Spot |
| 7 | Tomato__Tomato_YellowLeaf__Curl_Virus |
| 8 | Tomato__Tomato_mosaic_virus |
| 9 | Tomato_healthy |

## Particiones

El split es estratificado 80/10/10 con `random_state=42`. Se generó una vez y se guardó en `split.csv` para que todas las corridas usen exactamente las mismas imágenes.

| Partición | Imágenes |
|---|---|
| train | 12.808 |
| validation | 1.601 |
| test | 1.602 |
| total | 16.011 |

`split.csv` tiene las columnas `filepath`, `label` y `split`. `filepath` es relativo a `data/PlantVillage` y tiene el formato `Clase/archivo.JPG`.

## Estructura local

```
data/
  class_names.json
  split.csv
  PlantVillage/
    Tomato_Bacterial_spot/
    Tomato_Early_blight/
    ...
```

La carpeta `data/PlantVillage` no se versiona porque pesa cerca de 371 MB. Para reproducir el trabajo se descarga el dataset desde Kaggle y se descomprime en `data/PlantVillage`.

## Uso

El dataset se usa solo con fines académicos. Las imágenes de laboratorio no representan las condiciones de una huerta o de un cultivo real. Cualquier uso fuera del contexto del curso requiere una validación en campo y una revisión de las condiciones de captura.
