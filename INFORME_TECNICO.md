# Informe técnico EP3

## Pipeline integrado, regularización y comparación MLP vs CNN para patologías en tomate

**Asignatura:** TLY1102 Técnicas Avanzadas de Machine Learning I  
**Sección:** 001D  
**Integrantes:** Matías Gutiérrez, David Larenas, Victor Mena  
**Entrega:** Evaluación Parcial 3  
**Fecha:** Noviembre de 2026

---

## 1. Resumen ejecutivo

La CNN final alcanzó 94,4% de accuracy y 93,0% de F1 macro en el conjunto de prueba. La MLP final alcanzó 69,2% y 65,1% en las mismas métricas. La búsqueda seleccionó una CNN con lr 1e-04, batch 32, dropout conv 0,00, dropout denso 0,20, filtros [32, 64, 128], L2 0e+00. La CNN usa 4.290.378 parámetros frente a 12.626.762 de la MLP y generaliza mejor. El modelo se recomienda como apoyo a la priorización de casos, con revisión humana.

El trabajo se organiza en un pipeline de nueve etapas, dos arquitecturas comparables y una búsqueda de hiperparámetros con selección por F1 macro de validación. El conjunto de prueba se reserva para la evaluación final. La comparación incluye métricas de desempeño, generalización, costo computacional, auditoría de sesgos y explicabilidad.

## 2. Descripción del problema de negocio

El tomate es uno de los cultivos más relevantes de la agricultura intensiva y familiar. Las enfermedades foliares reducen el rendimiento y obligan a un diagnóstico oportuno, que en terreno depende de personal especializado. Una herramienta de apoyo visual puede priorizar casos y reducir el tiempo entre la captura de la imagen y la revisión técnica.

El problema de negocio no es reemplazar al agrónomo. Es construir un clasificador reproducible que distinga diez condiciones foliares del tomate con un desempeño suficiente para apoyar la priorización de casos. La decisión final sigue siendo humana, sobre todo cuando el modelo se equivoca entre enfermedades con síntomas parecidos.

El dataset disponible contiene imágenes de laboratorio, con una hoja centrada y un fondo uniforme. Esa condición facilita el aprendizaje, pero no representa una huerta real. El informe y la aplicación deben explicitar ese límite. Cualquier uso fuera del contexto académico requiere una validación en campo.

## 3. Objetivos

**Objetivo general.** Construir y comparar dos soluciones de aprendizaje profundo, una MLP y una CNN, para clasificar diez clases de patologías del tomate, usando un pipeline integrado, reproducible y auditable.

**Objetivos específicos.**

1. Reutilizar el split estratificado de la EP2 y mantener las mismas diez clases, el mismo orden y la misma resolución de entrada.
2. Implementar un pipeline de nueve etapas con funciones reutilizables, desde la carga hasta la auditoría.
3. Incorporar regularización con Dropout, Batch Normalization, L2, aumento de datos y parada temprana.
4. Optimizar hiperparámetros con experimentación controlada, primero un parámetro a la vez y luego una búsqueda aleatoria con presupuesto fijo.
5. Evaluar con accuracy, precision, recall, F1 macro, F1 ponderado, ROC-AUC, matrices de confusión y curvas de aprendizaje.
6. Auditar sesgos de entrada, de sistema y de aplicación, e identificar riesgos para el uso real.
7. Aplicar Grad-CAM a la CNN y LIME a la MLP para comparar la interpretabilidad de ambas arquitecturas.
8. Seleccionar un modelo final con evidencia experimental y justificar la decisión.

## 4. KPIs

| KPI | Definición | Meta |
|---|---|---|
| Accuracy | Proporción de aciertos sobre el total de imágenes de prueba | Reportar, no usar como único criterio |
| F1 macro | Promedio no ponderado del F1 por clase | Mayor o igual a 0,90 en la CNN final |
| F1 ponderado | F1 ponderado por el soporte de cada clase | Cercano al F1 macro, sin ocultar clases minoritarias |
| ROC-AUC | Capacidad de separación entre clases, promedio macro | Mayor o igual a 0,98 |
| Brecha train-test | Diferencia entre accuracy de entrenamiento y de prueba | Menor o igual a 0,10 |
| Tiempo de inferencia | Milisegundos por imagen en la GPU local | Menor o igual a 50 ms |
| Parámetros | Cantidad de parámetros entrenables | Reportar el costo de cada arquitectura |

El KPI principal es el F1 macro. El dataset está desbalanceado y la accuracy puede ocultar el desempeño en las clases minoritarias. El F1 ponderado se reporta como complemento, junto con el soporte de cada clase. La brecha train-test y el tiempo de inferencia permiten juzgar la generalización y el costo de uso.

## 5. Fuentes de datos

| Clase                                       |   train |   val |   test |   total |
|:--------------------------------------------|--------:|------:|-------:|--------:|
| Tomato_Bacterial_spot                       |   1,701 |   213 |    213 |   2,127 |
| Tomato_Early_blight                         | 800     |   100 |    100 |   1     |
| Tomato_Late_blight                          |   1,527 |   191 |    191 |   1,909 |
| Tomato_Leaf_Mold                            | 762     |    95 |     95 | 952     |
| Tomato_Septoria_leaf_spot                   |   1,417 |   177 |    177 |   1,771 |
| Tomato_Spider_mites_Two_spotted_spider_mite |   1,341 |   167 |    168 |   1,676 |
| Tomato__Target_Spot                         |   1,123 |   140 |    141 |   1,404 |
| Tomato__Tomato_YellowLeaf__Curl_Virus       |   2,566 |   321 |    321 |   3,208 |
| Tomato__Tomato_mosaic_virus                 | 298     |    38 |     37 | 373     |
| Tomato_healthy                              |   1,273 |   159 |    159 |   1,591 |

Usamos el subconjunto de tomate de PlantVillage, disponible en Kaggle como `emmarex/plantdisease`. Las imágenes originales son de 256 x 256 píxeles y el modelo las recibe a 128 x 128. El split es estratificado 80/10/10 con `random_state=42` y se guardó en `data/split.csv` para que todas las corridas usen exactamente las mismas imágenes.

La clase mayoritaria es `Tomato__Tomato_YellowLeaf__Curl_Virus` y la minoritaria es `Tomato__Tomato_mosaic_virus`. La razón entre ambas es cercana a 8,6 a 1. Esa diferencia es el primer foco de la auditoría.

## 6. Preparación y análisis exploratorio

Revisamos la cantidad de imágenes por clase y por partición, el tamaño y el modo de color, la existencia de archivos faltantes y la duplicación exacta. Todas las imágenes se pudieron leer, tienen tres canales y no se encontraron duplicados exactos. El split conserva las diez clases en train, validation y test.

![Balance de clases por partición](images/balance_clases.png)

La distribución se mantiene en las tres particiones por el split estratificado. La clase sana y la clase de virus del rizado amarillo concentran la mayor cantidad de imágenes. Las clases de mildiu y mosaico tienen menos ejemplos. El desbalance no se corrigió con sobremuestreo ni con pesos de clase, para no alterar la comparación con la EP2. El efecto se mide en la auditoría.

![Muestras de las diez clases](images/muestras_clases.png)

Las imágenes muestran una hoja centrada, iluminación controlada y fondo uniforme. Esa uniformidad explica por qué un modelo puede aprender la enfermedad y también por qué puede fallar en una foto de campo con fondo, sombra o varias hojas.

## 7. Metodología CRISP-DM

| Fase | Actividades | Entregable |
|---|---|---|
| Comprensión del negocio | Definir el problema, los objetivos y los KPIs | Este informe y la sección 1 del notebook |
| Comprensión de los datos | Revisar clases, tamaños, balance, duplicados y calidad | Tabla de datos y figuras del EDA |
| Preparación de los datos | Split estratificado, carga, redimensionado, normalización y aumento | `data/split.csv`, `cargar_datos`, `crear_dataset` |
| Modelado | MLP, CNN, regularización y búsqueda de hiperparámetros | `construir_modelo`, `optimizar_modelo` |
| Evaluación | Métricas, matrices, curvas, auditoría y XAI | `evaluar_metricas`, secciones 8 a 10 del notebook |
| Despliegue | Recomendación de uso y límites, sin puesta en producción | Modelo final y guía de uso |

No desplegamos el modelo. La fase de despliegue se limita a una recomendación documentada, porque el dataset es de laboratorio y no existe una validación externa. Presentar el modelo como un producto listo para campo sería una afirmación que los datos no sostienen.

## 8. Pipeline integrado

| Etapa | Función | Propósito |
|---|---|---|
| 1. Carga | `cargar_datos` | Leer el split, las clases y las rutas absolutas |
| 2. Análisis | `analizar_dataset` | Revisar balance, tamaños y muestras |
| 3. Preprocesamiento | `cargar_imagen`, `crear_dataset` | Decodificar, redimensionar, normalizar y crear lotes |
| 4. Construcción | `construir_modelo` | Armar la MLP o la CNN con regularización opcional |
| 5. Entrenamiento | `entrenar_modelo` | Entrenar con parada temprana y registrar el historial |
| 6. Regularización | `crear_aumento`, capas de Dropout, BatchNorm y L2 | Reducir el sobreajuste |
| 7. Evaluación | `evaluar_metricas`, `graficar_curvas` | Calcular métricas, matrices y curvas |
| 8. Auditoría | `prueba_estres`, análisis por clase | Medir sesgos y robustez |
| 9. Optimización | `optimizar_modelo` | Comparar configuraciones y elegir la mejor por validación |

El pipeline se implementa con funciones separadas y sin clases. Cada función recibe los datos, devuelve resultados y escribe sus artefactos en `models/` o `images/`. Esa separación permite reutilizar el código en la EP2 y en la EP3, y facilita la revisión por terceros.

## 9. Modelos MLP y CNN

| Modelo | Arquitectura | Regularización | Parámetros |
|---|---|---|---|
| MLP | Entrada 128 x 128 x 3, Flatten, Dense 512, 256 y 128, salida 10 | Dropout, BatchNorm, L2 y aumento según configuración | 12.626.762 |
| CNN | Tres bloques Conv2D de 32, 64 y 128 filtros, MaxPooling, Dense 128, salida 10 | Dropout, BatchNorm, L2 y aumento según configuración | 4.290.378 |

La MLP aplana la imagen y pierde la estructura espacial. La CNN comparte pesos en toda la imagen y conserva la noción de vecindad. Esa diferencia es la hipótesis principal de la comparación. La MLP se mantiene como línea base porque la EP2 la usó y porque permite medir cuánto aporta la convolución.

La CNN usa filtros de 3 x 3, activación ReLU, inicialización He y `padding='same'`. Cada bloque reduce la resolución con MaxPooling y aumenta la cantidad de filtros. La capa densa final tiene 128 neuronas y la salida es softmax. La configuración base usa Dropout de 0,2 en las capas convolucionales y de 0,3 en la capa densa.

## 10. Regularización y aumento de datos

La mejor configuración de la ablación fue A3_batch_norm con un F1 macro de validación de 91,7%. La CNN sin regularización obtuvo 90,8%, una diferencia de 1,0 puntos porcentuales. La técnica que más aportó fue A3_batch_norm. La brecha entre train y validación de la mejor variante fue de 6,5 puntos.

| Experimento           | F1 macro val   | Accuracy val   | Brecha val   | Parámetros   | Tiempo   |
|:----------------------|:---------------|:---------------|:-------------|:-------------|:---------|
| A3_batch_norm         | 91,7%          | 93,2%          | 6,5 pp       | 4,290,378    | 3,3 min  |
| A1_sin_regularizacion | 90,8%          | 92,1%          | 7,6 pp       | 4,288,970    | 3,9 min  |
| A2_dropout            | 90,6%          | 91,8%          | 1,0 pp       | 4,288,970    | 7,7 min  |
| A4_l2                 | 89,7%          | 91,0%          | 7,2 pp       | 4,288,970    | 3,0 min  |
| A5_aumento            | 78,7%          | 80,3%          | 11,4 pp      | 4,288,970    | 3,0 min  |
| A6_mlp_base           | 47,5%          | 60,0%          | -6,6 pp      | 25,331,850   | 2,7 min  |

![F1 macro de la ablación](images/ablacion_f1.png)

El aumento de datos aplica volteo horizontal, rotación de 0,08, zoom de 0,10, traslación de 0,08 y contraste de 0,10. No usamos volteo vertical porque una hoja invertida no es una condición real de captura. No usamos cambios de color agresivos porque el color es una señal diagnóstica y no queremos destruirla.

![Ejemplos de aumento de datos](images/aumento_datos.png)

La parada temprana monitorea `val_loss` con paciencia 5 y restaura los mejores pesos. El Dropout se activa solo durante el entrenamiento. Batch Normalization se aplica antes de la activación. L2 se aplica a los kernels convolucionales y a las capas densas con un factor pequeño.

## 11. Optimización de hiperparámetros

El mejor resultado del screening fue B5_dropout_dense_0.2 con un F1 macro de validación de 93,8%. La base regularizada obtuvo 68,1%. El cambio que más redujo el desempeño fue B4_dropout_conv_0.1. El screening permitió descartar los cambios que no aportaban antes de combinar hiperparámetros.

| Experimento           | F1 macro val   | Accuracy val   | Brecha val   | Parámetros   | Tiempo   |
|:----------------------|:---------------|:---------------|:-------------|:-------------|:---------|
| B5_dropout_dense_0,2  | 93,8%          | 94,9%          | 5,0 pp       | 4,290,378    | 3,3 min  |
| B6_filtros_64_128_256 | 93,7%          | 94,8%          | 5,1 pp       | 8,763,146    | 5,9 min  |
| B2_lr_1e-3            | 93,3%          | 94,1%          | 5,4 pp       | 4,290,378    | 3,7 min  |
| B7_l2_1e-5            | 92,7%          | 94,1%          | 5,8 pp       | 4,290,378    | 3,4 min  |
| B3_batch_64           | 92,1%          | 93,8%          | 6,2 pp       | 4,290,378    | 3,5 min  |
| B1_bn_base            | 91,5%          | 93,1%          | 6,4 pp       | 4,290,378    | 3,2 min  |
| B8_aumento            | 75,4%          | 78,0%          | 17,4 pp      | 4,290,378    | 6,5 min  |
| B4_dropout_conv_0,1   | 68,1%          | 76,8%          | 22,8 pp      | 4,290,378    | 3,4 min  |

![F1 macro del screening](images/screening_f1.png)

Descartamos GridSearchCV porque obliga a envolver el modelo de Keras y multiplica el costo por cada fold de validación cruzada. Descartamos la optimización bayesiana porque requiere una librería adicional y con ocho evaluaciones no compensa la complejidad. Usamos un screening univariado para conocer la dirección de cada hiperparámetro y luego un Random Search con presupuesto fijo.

La mejor configuración de la búsqueda fue C1_mejor_screening con un F1 macro de validación de 93,9%. Quedó con tasa de aprendizaje 1e-04, batch 32, dropout convolucional 0,00 y dropout denso 0,20. Superó a la mejor del screening por 0,1 puntos porcentuales.

| Experimento        |     LR |   Batch | Drop. conv   | Drop. denso   | Filtros    |     L2 | F1 macro val   | Brecha val   |
|:-------------------|-------:|--------:|:-------------|:--------------|:-----------|-------:|:---------------|:-------------|
| C1_mejor_screening | 0,0001 |      32 | 0,00         | 0,20          | 32-64-128  | 0      | 93,9%          | 4,8 pp       |
| C5_aleatorio       | 0,0001 |      32 | 0,01         | 0,25          | 64-128-256 | 0      | 92,4%          | 5,5 pp       |
| C7_aleatorio       | 0,0001 |      64 | 0,03         | 0,20          | 32-64-128  | 0,0001 | 89,5%          | 8,0 pp       |
| C3_aleatorio       | 0,0002 |      32 | 0,15         | 0,24          | 64-128-256 | 1e-05  | 87,8%          | 8,5 pp       |
| C4_aleatorio       | 0,0001 |      32 | 0,13         | 0,25          | 32-64-128  | 0,0001 | 78,1%          | 13,5 pp      |
| C2_aleatorio       | 7e-05  |      64 | 0,09         | 0,26          | 32-64-128  | 0,0001 | 70,6%          | 19,2 pp      |
| C6_aleatorio       | 0,0002 |      32 | 0,18         | 0,23          | 32-64-128  | 0,0001 | 69,9%          | 17,2 pp      |
| C8_aleatorio       | 0,0002 |      32 | 0,09         | 0,06          | 64-128-256 | 1e-05  | 69,9%          | 18,3 pp      |

![F1 macro de la búsqueda aleatoria](images/busqueda_f1.png)

El espacio de búsqueda cubre la tasa de aprendizaje en escala logarítmica entre 5e-5 y 1e-3, el tamaño de lote entre 32 y 64, el Dropout convolucional entre 0,0 y 0,2, el Dropout denso entre 0,0 y 0,3, dos configuraciones de filtros, L2 en 0, 1e-5 y 1e-4 y el aumento de datos como variable binaria. La selección se hace por F1 macro de validación. El test no se usa para elegir.

## 12. Resultados experimentales

La CNN final obtuvo 94,4% de accuracy y 93,0% de F1 macro en test, con una brecha train-test de 5,5 puntos. La MLP final obtuvo 69,2% de accuracy y 65,1% de F1 macro, con una brecha de 18,5 puntos. La diferencia de 27,9 puntos de F1 macro y la menor variación entre semillas de la CNN respaldan la elección de la CNN.

| Modelo    | Accuracy   | F1 macro   | F1 ponderado   |   ROC-AUC | Parámetros   | Entrenamiento   | Inferencia   | Brecha test   | F1 media   | F1 desviación   |
|:----------|:-----------|:-----------|:---------------|----------:|:-------------|:----------------|:-------------|:--------------|:-----------|:----------------|
| CNN final | 94,4%      | 93,0%      | 94,4%          |     0,997 | 4,290,378    | 3,4 min         | 6,2 ms       | 5,5 pp        | 93,0%      | 0,1 pp          |
| MLP final | 69,2%      | 65,1%      | 69,2%          |     0,964 | 12,626,762   | 1,5 min         | 4,6 ms       | 18,5 pp       | 57,9%      | 6,8 pp          |

![Curvas de la CNN final](images/curvas_cnn_final_semilla_42.png)

![Matriz de confusión de la CNN final](images/matriz_cnn_final_semilla_42.png)

La CNN final se entrena con la mejor configuración y tres semillas. La semilla 42 se guarda como modelo final. Las semillas 43 y 44 se usan para reportar la variabilidad. La MLP se entrena con su mejor configuración y las mismas tres semillas.

## 13. Auditoría de sesgos

Las clases con menor recall fueron Tomato_Early_blight (80,0%), Tomato_Late_blight (90,6%), Tomato__Target_Spot (92,2%). La diferencia entre F1 ponderado y F1 macro fue de 1,5 puntos, lo que confirma que las clases frecuentes pesan más. La CNN clasificó 2 casos sanos como enfermedad. En la prueba de estrés, el F1 macro cayó a 67,7% en el escenario contraste_bajo.

| Clase                                       |   Precision |   Recall |    F1 |   Soporte |
|:--------------------------------------------|------------:|---------:|------:|----------:|
| Tomato_Bacterial_spot                       |       0,986 |    0,995 | 0,991 |       213 |
| Tomato_Early_blight                         |       0,808 |    0,800 | 0,804 |       100 |
| Tomato_Late_blight                          |       0,911 |    0,906 | 0,908 |       191 |
| Tomato_Leaf_Mold                            |       0,899 |    0,937 | 0,918 |        95 |
| Tomato_Septoria_leaf_spot                   |       0,970 |    0,927 | 0,948 |       177 |
| Tomato_Spider_mites_Two_spotted_spider_mite |       0,951 |    0,923 | 0,937 |       168 |
| Tomato__Target_Spot                         |       0,890 |    0,922 | 0,906 |       141 |
| Tomato__Tomato_YellowLeaf__Curl_Virus       |       0,981 |    0,991 | 0,986 |       321 |
| Tomato__Tomato_mosaic_virus                 |       0,875 |    0,946 | 0,909 |        37 |
| Tomato_healthy                              |       1,000 |    0,987 | 0,994 |       159 |

![Confianza de la CNN en aciertos y errores](images/confianza_cnn.png)

| Transformación   | Accuracy   | F1 macro   |
|:-----------------|:-----------|:-----------|
| brillo_bajo      | 79,5%      | 77,9%      |
| brillo_alto      | 88,3%      | 86,5%      |
| contraste_bajo   | 73,0%      | 67,7%      |
| contraste_alto   | 84,1%      | 82,6%      |

![F1 macro bajo cambios de brillo y contraste](images/prueba_estres.png)

La auditoría se organiza en tres niveles. El sesgo de entrada incluye el desbalance de clases, la baja representación de mildiu y mosaico y el origen único de laboratorio. El sesgo de sistema incluye el uso de accuracy como métrica única, la confusión entre Early blight y Late blight y la diferencia entre F1 macro y F1 ponderado. El sesgo de aplicación incluye el uso en campo, la diferencia entre una hoja aislada y una planta completa y el riesgo de falsos negativos en la clase sana.

Las medidas de mitigación propuestas son aumentar la representación de las clases minoritarias, validar con imágenes de campo, mantener una revisión humana de los casos de baja confianza, monitorear el desempeño por clase y reentrenar cuando cambie la población de imágenes. Ninguna de estas medidas se implementó en esta entrega. Quedan como trabajo futuro.

## 14. Explicabilidad

Grad-CAM permite revisar las regiones que sostienen la predicción de la CNN. En los ejemplos revisados, el mapa se concentra en la hoja y, en varios casos, en la zona de la lesión. En los errores, la atención se dispersa hacia el fondo o hacia tejido sano. LIME en la MLP resalta regiones más fragmentadas, consistente con la pérdida de estructura espacial al aplanar la imagen. Las dos técnicas se usan como auditoría cualitativa y no como prueba de causalidad.

![Grad-CAM en una imagen por clase](images/gradcam_clases.png)

![Grad-CAM en errores de la CNN](images/gradcam_errores.png)

![LIME en la MLP](images/lime_mlp.png)

Grad-CAM muestra las regiones de la imagen que más influyen en la clase predicha. En la CNN, el mapa se calcula sobre la activación de la última capa convolucional y se superpone a la imagen original. LIME aproxima el modelo con perturbaciones locales y resalta las regiones que sostienen la predicción de la MLP.

Las dos técnicas no prueban causalidad. Sirven para revisar si el modelo mira la hoja o el fondo, si las confusiones tienen una explicación visual y si la MLP y la CNN usan señales distintas. Esa revisión es parte de la auditoría, no un reemplazo de las métricas.

## 15. Comparación MLP vs CNN

La CNN superó a la MLP por 27,9 puntos de F1 macro en test. Usa 2,9 veces menos parámetros. El entrenamiento de la CNN tardó 3,4 min y el de la MLP 1,5 min. La MLP conserva una ventaja en simplicidad y tiempo de inferencia. Por desempeño y generalización, recomendamos la CNN.

| Modelo    | Accuracy   | F1 macro   | F1 ponderado   |   ROC-AUC | Parámetros   | Entrenamiento   | Inferencia   | Brecha test   | F1 media   | F1 desviación   |
|:----------|:-----------|:-----------|:---------------|----------:|:-------------|:----------------|:-------------|:--------------|:-----------|:----------------|
| CNN final | 94,4%      | 93,0%      | 94,4%          |     0,997 | 4,290,378    | 3,4 min         | 6,2 ms       | 5,5 pp        | 93,0%      | 0,1 pp          |
| MLP final | 69,2%      | 65,1%      | 69,2%          |     0,964 | 12,626,762   | 1,5 min         | 4,6 ms       | 18,5 pp       | 57,9%      | 6,8 pp          |

![Comparación de accuracy y F1 macro](images/comparacion_modelos.png)

| Criterio | MLP | CNN | Comentario |
|---|---|---|---|
| Desempeño | Línea base | Mejor F1 macro | La convolución aprovecha la estructura espacial |
| Generalización | Brecha moderada | Brecha controlada | La regularización reduce la brecha |
| Parámetros | Más parámetros | Menos parámetros | La CNN comparte pesos |
| Tiempo de entrenamiento | Menor | Mayor | La CNN es más costosa por época |
| Tiempo de inferencia | Menor | Mayor | Ambas son rápidas para una imagen |
| Interpretabilidad | LIME | Grad-CAM | La CNN tiene una explicación espacial más natural |
| Prueba de estrés | No aplicada en esta entrega | Evaluada con cuatro cambios de brillo y contraste | La prueba se aplicó solo a la CNN |

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| Mantener tomate y diez clases | Cambiar de dataset | Conserva la trazabilidad con la EP1 y la EP2 |
| Usar 128 x 128 | Usar 256 x 256 | Reduce el costo y mantiene la comparación con la EP2 |
| Split 80/10/10 estratificado | Split aleatorio | Conserva las proporciones de clases |
| F1 macro como KPI | Solo accuracy | El dataset está desbalanceado |
| CNN como modelo principal | Solo MLP | La convolución aprovecha la estructura espacial |
| Random Search | GridSearchCV | Menor costo con un presupuesto fijo |
| Random Search | Búsqueda bayesiana | No se justifica con ocho evaluaciones |
| Tres semillas | Una semilla | Permite reportar variabilidad |
| Test solo al final | Ajustar con test | Evita fuga de información |
| Sin pesos de clase | Con pesos de clase | Mantiene la comparación y deja el sesgo a la auditoría |
| Sin transfer learning | Con transfer learning | El foco es la comparación y el pipeline |
| Grad-CAM para la CNN | SHAP para la CNN | Grad-CAM es espacial y está alineado con el curso |
| LIME para la MLP | Grad-CAM para la MLP | La MLP no tiene capas convolucionales |
| No desplegar | Poner en producción | No hay validación de campo |

La recomendación es usar la CNN como modelo principal y mantener la MLP como línea base y alternativa liviana. La CNN no se recomienda como diagnóstico autónomo. Se recomienda como apoyo a la priorización de casos, con revisión humana y con un monitoreo del desempeño por clase.

## 16. Limitaciones y trabajo futuro

1. El dataset es de laboratorio. No hay imágenes de campo, con fondo, sombra, varias hojas o distintas cámaras.
2. El test es una partición interna del mismo dataset. No hay un conjunto externo de validación.
3. El desbalance se documenta, pero no se corrige con pesos de clase ni con remuestreo.
4. La comparación usa tres semillas. Es suficiente para reportar variabilidad, no para un estudio estadístico exhaustivo.
5. El modelo clasifica una hoja aislada. No detecta la hoja dentro de una planta ni segmenta la lesión.
6. No se calibraron las probabilidades. La confianza se usa como señal de revisión, no como probabilidad clínica.
7. No se hizo una validación con agrónomos. La utilidad real requiere una evaluación con usuarios.
8. No se implementó un despliegue. El release contiene los modelos y los artefactos, no un servicio.

El trabajo futuro incluye recolectar imágenes de campo, evaluar la calibración, probar pesos de clase y transferencia desde modelos preentrenados, medir el desempeño con agrónomos y construir una interfaz de revisión que muestre la predicción, la confianza y el mapa de Grad-CAM.

## 17. Conclusiones

El pipeline permitió comparar las dos arquitecturas con el mismo split y las mismas condiciones. La CNN fue el mejor modelo por F1 macro y generalización, con 93,0% frente a 65,1% de la MLP. La MLP quedó como línea base y alternativa liviana. La limitación principal es el dataset de laboratorio y la falta de validación externa.

El pipeline quedó documentado y reproducible. El split, las semillas, las configuraciones y las métricas están en el repositorio. Cualquier persona puede repetir la corrida y comparar sus resultados con los de este informe. Esa trazabilidad es el aporte principal del trabajo, más allá de la diferencia entre las dos arquitecturas.

## 18. Referencias

1. Hughes, D. y Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.
2. Prince, S. J. D. (2023). Understanding Deep Learning. MIT Press.
3. Bishop, C. M. y Bishop, H. (2023). Deep Learning: Foundations and Concepts. Springer.
4. Goodfellow, I., Bengio, Y. y Courville, A. (2016). Deep Learning. MIT Press.
5. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
6. Ribeiro, M. T., Singh, S. y Guestrin, C. (2016). Why Should I Trust You? Explaining the Predictions of Any Classifier. KDD.
7. Chollet, F. (2021). Deep Learning with Python. Manning.
8. Wirth, R. y Hipp, J. (2000). CRISP-DM: Towards a Standard Process Model for Data Mining. Proceedings of the 4th International Conference on the Practical Application of Knowledge Discovery and Data Mining.

## 19. Anexos

### A. Estructura del repositorio

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

### B. Reproducibilidad

La corrida usa `SEED=42`, el split de `data/split.csv`, imágenes de 128 x 128 y un máximo de 40 épocas con parada temprana. La búsqueda de hiperparámetros se ejecuta en el notebook y los resultados se guardan en `models/experimentos.csv`. Los modelos finales se publican en el release `v2.0` y no se versionan en git.

### C. Enlaces

- Repositorio: https://github.com/mtgutierrezh/tly1102-tomate
- Release: https://github.com/mtgutierrezh/tly1102-tomate/releases/tag/v2.0
- Notebook: `notebooks/EP3_Gutierrez_Larenas_Mena.ipynb`
- Defensa: `DEFENSA_EP3.md`
- Presentación: `docs/presentacion.md`
