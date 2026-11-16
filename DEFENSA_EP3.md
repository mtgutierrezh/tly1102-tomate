# Defensa EP3

## Guion de 10 minutos

### 0:00 a 1:00. Apertura

Buenos días. Somos Matías Gutiérrez, David Larenas y Victor Mena. Presentamos la Evaluación Parcial 3 de Técnicas Avanzadas de Machine Learning I. El trabajo compara una MLP y una CNN para clasificar diez patologías foliares del tomate. La pregunta que guía la presentación es por qué elegimos la CNN y qué evidencia sostiene esa decisión.

### 1:00 a 2:00. Problema y KPIs

El problema de negocio es apoyar la priorización de casos en tomate. No buscamos reemplazar al agrónomo. Buscamos un clasificador reproducible que ordene casos y ayude a decidir cuáles revisar primero. El KPI principal es el F1 macro, porque el dataset está desbalanceado y la accuracy puede ocultar las clases minoritarias. También reportamos accuracy, F1 ponderado, ROC-AUC, brecha train-test, tiempo de inferencia y cantidad de parámetros.

### 2:00 a 3:00. Datos y EDA

Usamos el subconjunto de tomate de PlantVillage, con 16.011 imágenes y diez clases. El split es estratificado 80/10/10 y está guardado en `split.csv`. La clase mayoritaria tiene 2.566 imágenes y la minoritaria 298, una razón cercana a 8,6 a 1. Las imágenes son de laboratorio, con hoja centrada y fondo uniforme. Ese detalle es una fortaleza para el aprendizaje y una limitación para el uso en campo.

### 3:00 a 4:30. Pipeline y modelos

El pipeline tiene nueve etapas, desde la carga hasta la auditoría. Cada etapa está en una función y deja artefactos en `models/` o `images/`. La MLP aplana la imagen y usa capas densas de 512, 256 y 128. La CNN usa tres bloques convolucionales de 32, 64 y 128 filtros, con MaxPooling, BatchNorm, Dropout y una capa densa final. La CNN tiene {{CNN_PARAMS}} parámetros y la MLP {{MLP_PARAMS}}. La diferencia clave es que la CNN conserva la estructura espacial y comparte pesos.

### 4:30 a 6:00. Regularización y optimización

Aplicamos Dropout, BatchNorm, L2, aumento de datos y parada temprana. La ablación mide el efecto de cada técnica por separado. El aumento incluye volteo horizontal, rotación, zoom, traslación y contraste. No usamos volteo vertical porque una hoja invertida no es una condición real. Para optimizar, hicimos un screening univariado y luego un Random Search con ocho configuraciones. Elegimos Random Search porque GridSearchCV multiplica el costo por cada fold y la búsqueda bayesiana no se justifica con ocho evaluaciones. La selección se hace por F1 macro de validación. El test no se toca.

### 6:00 a 8:00. Resultados, auditoría y XAI

La CNN final alcanza {{CNN_ACC}} de accuracy y {{CNN_F1}} de F1 macro. La MLP final alcanza {{MLP_ACC}} y {{MLP_F1}}. La CNN usa menos parámetros y tarda más por época. La auditoría muestra el desempeño por clase, la confusión entre Early blight y Late blight, la confianza en aciertos y errores y una prueba de estrés con cambios de brillo y contraste. Grad-CAM muestra que la CNN mira la hoja y no el fondo en la mayoría de los casos. LIME muestra las regiones que sostienen la predicción de la MLP. Las dos técnicas ayudan a revisar el modelo, pero no prueban causalidad.

### 8:00 a 9:00. Comparación y recomendación

La CNN gana en F1 macro, generalización y robustez. La MLP gana en tiempo de entrenamiento y simplicidad. Recomendamos la CNN como modelo principal y la MLP como línea base. No recomendamos un diagnóstico autónomo. Recomendamos un uso de apoyo, con revisión humana de los casos de baja confianza.

### 9:00 a 10:00. Limitaciones y cierre

La limitación principal es el dataset de laboratorio. No hay imágenes de campo ni validación externa. El test es una partición interna. No corregimos el desbalance con pesos de clase. El siguiente paso es recolectar imágenes de campo, calibrar las probabilidades y validar con agrónomos. Con eso, el pipeline que construimos queda listo para una evaluación más realista. Gracias.

## Cifras clave

| Dato | Valor |
|---|---|
| Imágenes totales | 16.011 |
| Train | 12.808 |
| Validation | 1.601 |
| Test | 1.602 |
| Clases | 10 |
| Razón de desbalance | 8,6 a 1 |
| Parámetros CNN | {{CNN_PARAMS}} |
| Parámetros MLP | {{MLP_PARAMS}} |
| Accuracy CNN | {{CNN_ACC}} |
| F1 macro CNN | {{CNN_F1}} |
| Accuracy MLP | {{MLP_ACC}} |
| F1 macro MLP | {{MLP_F1}} |
| Mejor configuración | {{MEJOR_CONFIG}} |
| Tiempo CNN | {{CNN_TIME}} |
| Tiempo MLP | {{MLP_TIME}} |
| Inferencia CNN | {{CNN_INF}} |
| Inferencia MLP | {{MLP_INF}} |

## Decisiones y alternativas

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
| Sin transfer learning | Con transfer learning | El foco es la comparación y el pipeline, no el máximo puntaje |
| Grad-CAM para la CNN | SHAP para la CNN | Grad-CAM es espacial y está alineado con el curso |
| LIME para la MLP | Grad-CAM para la MLP | La MLP no tiene capas convolucionales |
| No desplegar | Poner en producción | No hay validación de campo |

## Preguntas y respuestas

### Problema y datos

**1. ¿Por qué tomate y no otro dataset?**  
Porque la EP1 y la EP2 usaron el mismo subconjunto y la comparación necesita continuidad. Cambiar de dataset habría roto la trazabilidad.

**2. ¿Por qué diez clases y no las 38 de PlantVillage?**  
Porque las diez clases de tomate son las que venimos trabajando desde la EP1. Agregar clases habría cambiado el problema y el balance.

**3. ¿Por qué 128 x 128 y no 256 x 256?**  
Porque reduce el costo de entrenamiento y mantiene la comparación con la EP2. Las imágenes originales son de 256 x 256 y se redimensionan antes del modelo.

**4. ¿Por qué 80/10/10?**  
Porque permite un test de 1.602 imágenes y una validación de 1.601, suficientes para medir el desempeño y ajustar sin tocar el test.

**5. ¿Por qué el split es estratificado?**  
Porque conserva la proporción de las diez clases en las tres particiones. Un split aleatorio podría dejar clases minoritarias con muy pocos ejemplos en test.

### Pipeline y arquitectura

**6. ¿Cuáles son las nueve etapas del pipeline?**  
Carga, análisis, preprocesamiento, construcción, entrenamiento, regularización, evaluación, auditoría y optimización. Cada etapa tiene una función y un artefacto.

**7. ¿Por qué usaron funciones y no clases?**  
Porque el pipeline es lineal y las funciones son más simples de leer y reutilizar. Las clases agregarían estado sin un beneficio claro para este tamaño de proyecto.

**8. ¿Cómo se conecta el pipeline con CRISP-DM?**  
Las primeras etapas cubren la comprensión y la preparación. El modelado y la evaluación cubren las fases centrales. La auditoría y la recomendación cubren la evaluación y el despliegue.

**9. ¿Por qué la CNN tiene menos parámetros que la MLP?**  
Porque comparte los pesos de los filtros en toda la imagen. La MLP conecta cada píxel con cada neurona y eso multiplica los parámetros.

**10. ¿Por qué usaron MaxPooling?**  
Porque reduce la resolución, disminuye el costo y hace que la representación sea menos sensible a pequeños desplazamientos.

### Regularización

**11. ¿Qué técnicas de regularización usaron?**  
Dropout, BatchNorm, L2, aumento de datos y parada temprana. La ablación mide el efecto de las cuatro primeras.

**12. ¿Por qué Dropout en las capas convolucionales y en la densa?**  
Porque el Dropout convolucional regulariza los filtros y el denso regulariza la combinación final. Usamos valores distintos porque la capa densa tiene más parámetros.

**13. ¿Por qué BatchNorm antes de la activación?**  
Porque normaliza la salida de la capa y estabiliza el entrenamiento. Es la ubicación que usa el material del curso y la que probamos en el pipeline.

**14. ¿Por qué L2 y no L1?**  
Porque L2 penaliza los pesos grandes de forma suave y es la opción habitual para redes profundas. L1 puede llevar muchos pesos a cero y no era el objetivo.

**15. ¿Por qué no usaron volteo vertical en el aumento?**  
Porque una hoja invertida no es una condición real de captura. El aumento debe parecerse a las variaciones que el modelo verá en uso.

### Optimización

**16. ¿Por qué Random Search y no GridSearchCV?**  
Porque GridSearchCV obliga a envolver Keras y multiplica el costo por cada fold. Random Search cubre el espacio con un presupuesto fijo y es reproducible.

**17. ¿Por qué no usaron optimización bayesiana?**  
Porque requiere una librería adicional y con ocho evaluaciones no compensa la complejidad. La ganancia esperada no justificaba el costo.

**18. ¿Cómo eligieron la mejor configuración?**  
Por F1 macro de validación. El test se reserva para la evaluación final. También revisamos la brecha entre train y validación.

**19. ¿Por qué el screening es univariado?**  
Porque permite ver la dirección de cada hiperparámetro con un cambio a la vez. Es una etapa exploratoria antes de combinar valores.

**20. ¿Por qué tres semillas?**  
Porque una sola semilla no permite distinguir una mejora real de la variabilidad del entrenamiento. Tres semillas dan una idea de la media y la dispersión.

### Métricas y evaluación

**21. ¿Por qué F1 macro y no accuracy?**  
Porque el dataset está desbalanceado. El F1 macro da el mismo peso a cada clase y evita que la clase mayoritaria domine el resultado.

**22. ¿Qué diferencia hay entre F1 macro y F1 ponderado?**  
El macro promedia por clase y el ponderado pesa por el soporte. Si el ponderado es mucho mayor, el modelo está mejor en las clases frecuentes.

**23. ¿Por qué reportan ROC-AUC?**  
Porque mide la capacidad de separación entre clases y no depende de un umbral fijo. Es un complemento de las métricas de clasificación.

**24. ¿Qué es la brecha train-test?**  
Es la diferencia entre la accuracy de entrenamiento y la de prueba. Una brecha grande sugiere sobreajuste. Una brecha pequeña con accuracy baja sugiere underfitting.

**25. ¿Por qué no usaron el test para ajustar?**  
Porque el test debe medir el desempeño final sobre datos no vistos. Usarlo para elegir sería una fuga de información y las métricas dejarían de ser válidas.

### Sesgos y ética

**26. ¿Cuál es el sesgo de entrada principal?**  
El desbalance de clases y el origen único de laboratorio. La clase mayoritaria tiene 2.566 imágenes y la minoritaria 298.

**27. ¿Cuál es el sesgo de sistema principal?**  
El uso de accuracy como métrica única y la confusión entre Early blight y Late blight. Por eso reportamos F1 macro y la matriz de confusión.

**28. ¿Cuál es el sesgo de aplicación principal?**  
Usar el modelo en campo cuando fue entrenado con imágenes de laboratorio. Una foto con fondo, sombra o varias hojas puede cambiar la predicción.

**29. ¿Qué mitigaciones proponen?**  
Más datos de las clases minoritarias, imágenes de campo, revisión humana de los casos de baja confianza y monitoreo por clase.

**30. ¿El modelo puede usarse como diagnóstico autónomo?**  
No. Es una herramienta de apoyo. La decisión final debe ser de una persona con criterio agronómico.

### Explicabilidad

**31. ¿Por qué Grad-CAM para la CNN?**  
Porque usa las activaciones convolucionales y muestra las regiones espaciales que sostienen la predicción. Es adecuado para una CNN.

**32. ¿Por qué LIME para la MLP?**  
Porque la MLP no tiene capas convolucionales y LIME puede explicar un modelo tabular o una imagen aplanada a partir de perturbaciones locales.

**33. ¿Grad-CAM prueba que el modelo es correcto?**  
No. Muestra las regiones que influyen en la predicción, no una relación causal. Puede haber sesgos en la explicación.

**34. ¿Qué revisaron en los mapas de Grad-CAM?**  
Si el modelo mira la hoja o el fondo, si las confusiones tienen una explicación visual y si los aciertos se sostienen en la lesión.

**35. ¿Qué límite tiene LIME?**  
La explicación es local y depende de las perturbaciones. No representa el comportamiento global del modelo.

### Comparación y despliegue

**36. ¿Por qué la CNN es el modelo recomendado?**  
Porque obtiene mejor F1 macro, mejor generalización y mejor robustez ante cambios de brillo y contraste. Además tiene menos parámetros.

**37. ¿En qué gana la MLP?**  
En tiempo de entrenamiento y simplicidad. Es una línea base útil y una alternativa liviana.

**38. ¿Por qué no usaron transfer learning?**  
Porque el foco es la comparación entre arquitecturas y el pipeline. Transfer learning habría cambiado la comparación y el alcance del curso.

**39. ¿Qué falta para poner el modelo en producción?**  
Validación con imágenes de campo, calibración de probabilidades, revisión con agrónomos, monitoreo y un plan de actualización.

**40. ¿Cuál es el aporte principal del trabajo?**  
Un pipeline reproducible, con split fijo, semillas, experimentos, auditoría y explicabilidad. La decisión entre MLP y CNN queda respaldada por evidencia y no solo por una corrida.

## Cierre de un minuto

La CNN es el modelo recomendado porque aprovecha la estructura espacial, usa menos parámetros y generaliza mejor. La MLP queda como línea base. El pipeline es reproducible y deja trazabilidad de cada decisión. La limitación principal es el dataset de laboratorio, así que el siguiente paso es validar con imágenes de campo. Gracias.
