# Supuestos y Limitaciones

## Supuestos

1. **Representatividad del dataset**: El dataset GRAPE Leaf Diseases representa una muestra inicial de condiciones visibles en hojas de uva, aunque no cubre toda la variabilidad real de campo.

2. **Relevancia de clases**: Las cuatro clases del dataset (BlackRot, Esca, Healthy, LeafBlight) corresponden a condiciones visibles relevantes para el negocio de calidad y trazabilidad.

3. **Confiabilidad de etiquetas**: Las etiquetas entregadas se consideran confiables como base para una prueba de concepto, aunque se auditaron parcialmente.

4. **Aproximación al problema real**: Las imágenes de hojas de uva en condiciones controladas son una aproximación al tipo de problema visual que la empresa enfrenta en campo.

5. **Rol del modelo**: El modelo se usará como apoyo a la priorización de revisiones, no como reemplazo total del criterio técnico del equipo agronómico.

6. **Condiciones de captura**: Se asume que las imágenes en producción tendrán calidad y resolución similar a las del dataset de entrenamiento.

## Limitaciones

1. **Dataset pequeño**: Con 4,195 imágenes, el dataset es reducido para entrenar un modelo de detección robusto. Esto limita la capacidad de generalización.

2. **Desbalance de clases**: La clase Healthy tiene significativamente menos ejemplos (643) que Esca (1,383), lo que puede generar sesgo en las predicciones.

3. **Riesgo de sobreajuste**: El modelo puede memorizar patrones específicos del dataset en lugar de aprender características generales de las enfermedades.

4. **Diferencia con condiciones reales**: El dataset proviene de Kaggle (condiciones controladas). La operación real puede tener iluminación variable, fondos distintos, calidad de cámara diferente y oclusión por otras hojas.

5. **Sin validación con usuarios finales**: No se ha validado el modelo con el equipo técnico o agronómico que usará el sistema.

6. **Sin medición de impacto económico real**: No se ha medido cómo las alertas del sistema impactan en tiempo de revisión, calidad de decisión o costos operativos.

7. **Solo detección por imagen**: El sistema procesa imágenes individuales. No hay seguimiento temporal ni análisis de evolución de condiciones en el tiempo.

8. **Sin datos de campo real**: No se dispone de imágenes capturadas en las condiciones reales de operación de la empresa.

## Recomendaciones

1. Recolectar al menos 500-1000 imágenes propias con condiciones reales de campo
2. Balancear las clases mediante aumentación o recolección dirigida
3. Realizar validación cruzada con datos de campo
4. Involucrar al equipo técnico en la definición de criterios de alerta
5. Implementar un monitoreo de desempeño continuo en producción
