# Presentación EP3

Contenido para 12 diapositivas. Cada bloque se puede copiar a PowerPoint o Canva. Las figuras están en `images/`.

## Diapositiva 1. Portada

**Título:** Pipeline integrado y comparación MLP vs CNN para patologías en tomate

**Subtítulo:** Evaluación Parcial 3, TLY1102 Técnicas Avanzadas de Machine Learning I

**Integrantes:** Matías Gutiérrez, David Larenas, Victor Mena

**Idea clave:** Un pipeline reproducible para comparar dos arquitecturas con evidencia.

## Diapositiva 2. Problema y KPIs

- El diagnóstico foliar oportuno reduce pérdidas, pero depende de personal especializado.
- Buscamos apoyo a la priorización, no un diagnóstico autónomo.
- KPI principal: F1 macro, por el desbalance de clases.
- KPIs complementarios: accuracy, ROC-AUC, brecha train-test, tiempo de inferencia.

**Figura:** ninguna.

**Idea clave:** El problema se mide con F1 macro, no solo con accuracy.

## Diapositiva 3. Datos y EDA

- PlantVillage, subconjunto de tomate, 16.011 imágenes y 10 clases.
- Split estratificado 80/10/10: 12.808 train, 1.601 validation, 1.602 test.
- Desbalance de 8,6 a 1 entre la clase mayoritaria y la minoritaria.
- Imágenes de laboratorio, hoja centrada y fondo uniforme.

**Figura:** `images/balance_clases.png` y `images/muestras_clases.png`.

**Idea clave:** El dataset es limpio y controlado, pero no representa el campo.

## Diapositiva 4. Pipeline integrado

- Nueve etapas: carga, análisis, preprocesamiento, construcción, entrenamiento, regularización, evaluación, auditoría y optimización.
- Cada etapa tiene una función y deja artefactos en `models/` o `images/`.
- El split y las semillas están fijos para que la corrida sea reproducible.
- El pipeline se reutiliza de la EP2 y se amplía con auditoría y XAI.

**Figura:** tabla de etapas del informe.

**Idea clave:** El pipeline es el aporte principal, no solo el modelo.

## Diapositiva 5. Modelos MLP y CNN

- MLP: Flatten, Dense 512, 256 y 128, salida 10, 12.626.762 parámetros.
- CNN: Conv2D 32, 64 y 128, MaxPooling, Dense 128, salida 10, 4.290.378 parámetros.
- La CNN conserva la estructura espacial y comparte pesos.
- La MLP es la línea base y la alternativa liviana.

**Figura:** resumen de arquitecturas del notebook.

**Idea clave:** La convolución aprovecha la imagen, la MLP la aplana.

## Diapositiva 6. Regularización y aumento

- Dropout, BatchNorm, L2, aumento de datos y parada temprana.
- Aumento: volteo horizontal, rotación, zoom, traslación y contraste.
- Sin volteo vertical, porque una hoja invertida no es una condición real.
- La ablación mide el efecto de cada técnica por separado.

**Figura:** `images/ablacion_f1.png` y `images/aumento_datos.png`.

**Idea clave:** Cada técnica se justifica con una comparación, no por costumbre.

## Diapositiva 7. Optimización de hiperparámetros

- Screening univariado para conocer la dirección de cada hiperparámetro.
- Random Search con ocho configuraciones y presupuesto fijo.
- Espacio: tasa de aprendizaje, batch, dropouts, filtros y L2.
- Selección por F1 macro de validación. El test no se toca.

**Figura:** `images/screening_f1.png` y `images/busqueda_f1.png`.

**Idea clave:** Random Search cubre el espacio sin el costo de GridSearchCV.

## Diapositiva 8. Resultados

- CNN final: accuracy 94,4%, F1 macro 93,0%.
- MLP final: accuracy 69,2%, F1 macro 65,1%.
- La CNN usa menos parámetros y tarda más por época.
- La brecha en la mejor época es 5,4 puntos en la CNN y 15,7 en la MLP.

**Figura:** `images/curvas_cnn_final_semilla_42.png`, `images/matriz_cnn_final_semilla_42.png` y `images/comparacion_modelos.png`.

**Idea clave:** La CNN gana en F1 macro y generalización. El modelo final es la mejor época, no la última.

## Diapositiva 9. Auditoría de sesgos

- Entrada: desbalance 8,6 a 1 y origen único de laboratorio.
- Sistema: recall por clase, confusión Early blight y Late blight, confianza en errores.
- Aplicación: riesgo al usar imágenes de campo y falsos negativos en la clase sana.
- Mitigación: más datos, revisión humana, monitoreo y validación en campo.

**Figura:** `images/confianza_cnn.png` y `images/prueba_estres.png`.

**Idea clave:** El modelo es bueno en laboratorio, pero no está listo para campo.

## Diapositiva 10. Explicabilidad

- Grad-CAM en la CNN: regiones que sostienen la predicción.
- LIME en la MLP: perturbaciones locales sobre la imagen que recibe la red.
- Revisamos aciertos y errores para ver si el modelo mira la hoja o el fondo.
- Las explicaciones no prueban causalidad, pero ayudan a auditar.

**Figura:** `images/gradcam_clases.png`, `images/gradcam_errores.png` y `images/lime_mlp.png`.

**Idea clave:** La explicabilidad es parte de la auditoría, no un adorno.

## Diapositiva 11. Comparación y recomendación

- La CNN gana en F1 macro y generalización, la MLP en tiempo y simplicidad.
- La prueba de estrés se aplicó solo a la CNN, no a la MLP.
- Recomendamos la CNN como modelo principal y la MLP como línea base.
- No recomendamos un diagnóstico autónomo. Recomendamos apoyo con revisión humana.

**Figura:** `images/comparacion_modelos.png`.

**Idea clave:** La CNN es la mejor opción, con límites explícitos.

## Diapositiva 12. Conclusiones y limitaciones

- Pipeline reproducible, con split fijo, semillas y artefactos.
- La ablación mide el efecto de la regularización y el aumento.
- La limitación principal es el dataset de laboratorio y la falta de validación externa.
- Trabajo futuro: imágenes de campo, calibración y validación con agrónomos.

**Figura:** ninguna.

**Idea clave:** El trabajo deja una base sólida y un camino claro para el campo.
