# EP3. Pipeline integrado y comparación MLP vs CNN para patologías en tomate

**TLY1102 Técnicas Avanzadas de Machine Learning I, sección 001D**  
Matías Gutiérrez, David Larenas, Victor Mena

Este repositorio contiene la Evaluación Parcial 3. Reutiliza el dataset y el split de la EP2, incorpora regularización, búsqueda de hiperparámetros, auditoría de sesgos y explicabilidad. El objetivo es comparar una MLP y una CNN con un pipeline reproducible y elegir la mejor solución con evidencia.

## Resultados

| Modelo | Accuracy | F1 macro | F1 ponderado | ROC-AUC | Parámetros | Entrenamiento | Inferencia |
|---|---|---|---|---|---|---|---|
| CNN final | {{CNN_ACC}} | {{CNN_F1}} | {{CNN_F1W}} | {{CNN_AUC}} | {{CNN_PARAMS}} | {{CNN_TIME}} | {{CNN_INF}} |
| MLP final | {{MLP_ACC}} | {{MLP_F1}} | {{MLP_F1W}} | {{MLP_AUC}} | {{MLP_PARAMS}} | {{MLP_TIME}} | {{MLP_INF}} |

La CNN es el modelo recomendado. La MLP queda como línea base y alternativa liviana. Los modelos se publican en el release `v2.0`.

## Qué cambió respecto de la EP2

| Elemento | EP2 | EP3 |
|---|---|---|
| Dataset y split | Tomate, 10 clases, 80/10/10 | Se mantiene sin cambios |
| Arquitecturas | MLP base y CNN base | MLP y CNN con regularización |
| Aumento de datos | No | Volteo, rotación, zoom, traslación y contraste |
| Optimización | Configuración fija | Screening y Random Search |
| Métricas | Accuracy y F1 macro | Accuracy, F1 macro, F1 ponderado y ROC-AUC |
| Auditoría | Análisis de errores | Auditoría de entrada, sistema y aplicación |
| XAI | Grad-CAM | Grad-CAM y LIME |
| Documentación | README breve | README, informe, defensa y presentación |

## Pipeline

1. Carga del split y de las clases.
2. Análisis exploratorio y balance.
3. Preprocesamiento y aumento de datos.
4. Construcción de la MLP o la CNN.
5. Entrenamiento con parada temprana.
6. Regularización con Dropout, BatchNorm, L2 y aumento.
7. Evaluación con métricas, matrices y curvas.
8. Auditoría de sesgos y prueba de estrés.
9. Optimización y selección del modelo final.

El notebook desarrolla cada etapa. Las funciones están separadas y documentadas en celdas Markdown. Los artefactos se guardan en `models/` y las figuras en `images/`.

## Estructura

```
tly1102-tomate/
  README.md
  INFORME_TECNICO.md
  DEFENSA_EP3.md
  requirements.txt
  LICENSE
  data/
    README.md
    split.csv
    class_names.json
  notebooks/
    EP3_Gutierrez_Larenas_Mena.ipynb
  models/
    README.md
    experimentos.csv
    metricas_finales.json
    comparacion_final.csv
    historial/
    reportes/
  images/
  docs/
    presentacion.md
  legacy/
    EP2_Gutierrez_Larenas_Mena.ipynb
    README_EP2.md
    images/
    models/
```

## Cómo ejecutar

### Google Colab

1. Abrir el notebook en Colab.
2. Subir `PlantVillage Dataset.zip` a la raíz del repositorio en `/content/tly1102-tomate`.
3. Ejecutar todas las celdas. La primera celda instala `lime` si falta.
4. Usar una GPU T4 o superior. La corrida completa tarda cerca de 8 horas en la GPU local.

### Local

```bash
git clone https://github.com/mtgutierrezh/tly1102-tomate.git
cd tly1102-tomate
pip install -r requirements.txt
jupyter lab notebooks/EP3_Gutierrez_Larenas_Mena.ipynb
```

El dataset se descarga desde Kaggle como `emmarex/plantdisease` y se descomprime en `data/PlantVillage`. La carpeta no se versiona porque pesa cerca de 371 MB.

## Reproducibilidad

- `SEED=42` para el split y las corridas.
- `data/split.csv` fija las 16.011 imágenes en train, validation y test.
- `data/class_names.json` fija el orden de las diez clases.
- Las imágenes se redimensionan a 128 x 128 y se normalizan con `Rescaling(1/255)`.
- La búsqueda de hiperparámetros se ejecuta en el notebook y se registra en `models/experimentos.csv`.
- Los modelos finales se publican en el release `v2.0` y no se versionan en git.

## Entregables

| Archivo | Contenido |
|---|---|
| `notebooks/EP3_Gutierrez_Larenas_Mena.ipynb` | Pipeline completo, experimentos y figuras |
| `INFORME_TECNICO.md` | Informe con problema, KPIs, CRISP-DM y resultados |
| `DEFENSA_EP3.md` | Guion de 10 minutos y 40 preguntas |
| `docs/presentacion.md` | Contenido de 12 diapositivas |
| `models/metricas_finales.json` | Métricas finales y configuración |
| `models/experimentos.csv` | Registro de las 30 corridas |
| Release `v2.0` | Modelos, métricas, figuras y split |

## Limitaciones y ética

El dataset es de laboratorio, con hoja centrada y fondo uniforme. El test es una partición interna y no hay validación externa. El modelo no debe usarse como diagnóstico autónomo. Se recomienda como apoyo a la priorización de casos, con revisión humana y monitoreo del desempeño por clase. El desbalance se documenta y se audita, pero no se corrige con pesos de clase para mantener la comparación con la EP2.

## Enlaces

- Repositorio: https://github.com/mtgutierrezh/tly1102-tomate
- Release: https://github.com/mtgutierrezh/tly1102-tomate/releases/tag/v2.0
- Informe: `INFORME_TECNICO.md`
- Defensa: `DEFENSA_EP3.md`
- Presentación: `docs/presentacion.md`
- EP2: `legacy/README_EP2.md`
