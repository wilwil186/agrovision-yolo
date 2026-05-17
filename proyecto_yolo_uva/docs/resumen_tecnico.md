# Resumen Técnico

## 1. Problema de Negocio

La empresa busca explorar una solución basada en visión por computadora para apoyar procesos de calidad, trazabilidad y priorización de revisiones en campo. Actualmente las inspecciones visuales son manuales y están sujetas a variabilidad por experiencia del personal, volumen de trabajo y condiciones de observación.

## 2. Objetivo Técnico

Construir una prueba de concepto con YOLO que permita detectar condiciones visibles en hojas de uva, generar una predicción con nivel de confianza y convertir esa predicción en una recomendación operativa de revisión o priorización.

## 3. Dataset Recibido

- **Fuente**: GRAPE Leaf Diseases (Kaggle)
- **Formato**: YOLO object detection
- **Clases**: Grape__BlackRot, Grape__Esca, Grape__Healthy, Grape__LeafBlight
- **Particiones**: 3,358 imágenes train / 837 imágenes val
- **Anotaciones**: 4,415 bounding boxes en total

## 4. Auditoría de Datos

- Todas las imágenes tienen su etiqueta correspondiente
- Formato YOLO válido (class_id x_center y_center width height)
- Coordenadas normalizadas entre 0 y 1
- Sin errores estructurales en las etiquetas
- Clase Healthy con menos ejemplos (643) frente a Esca (1,383)

## 5. Preparación Realizada

- Copia del dataset a `data/processed/` con estructura YOLO estándar
- Creación de `dataset.yaml` con rutas locales corregidas
- Semilla fija (42) para reproducibilidad

## 6. Modelo Elegido

- **Arquitectura**: YOLOv8n (nano)
- **Enfoque**: Transfer learning con pesos preentrenados en COCO
- **Justificación**: Modelo ligero, rápido de entrenar, adecuado para dataset pequeño y posible despliegue en edge

## 7. Entrenamiento

- **Épocas**: 30 (con early stopping de 10)
- **Tamaño de imagen**: 640x640
- **Batch**: 8
- **Aumentación**: Flip horizontal, cambios de brillo/contraste, rotaciones leves

## 8. Métricas Obtenidas

- mAP50
- mAP50-95
- Precision
- Recall

## 9. Interpretación de Resultados

- Precision alta → cuando el sistema alerta, suele tener razón
- Recall alto → el sistema deja pasar pocos casos relevantes
- Falsos positivos → generan revisiones innecesarias
- Falsos negativos → pueden pasar condiciones importantes desapercibidas

## 10. Reglas de Alerta

| Nivel | Criterio | Acción |
|---|---|---|
| Alta prioridad | Confianza >= 0.70 + condición detectable | Revisión técnica prioritaria |
| Revisión manual | Confianza >= 0.40 + condición detectable | Validación humana |
| Monitoreo | Confianza < 0.40 + condición detectable | Guardar evidencia |
| Sin alerta | Sin detecciones relevantes | No priorizar |

## 11. Riesgos y Limitaciones

- Dataset pequeño (4,195 imágenes)
- Desbalance de clases (Healthy subrepresentada)
- Riesgo de sobreajuste por dataset controlado
- Diferencia entre dataset público y condiciones reales de campo
- Sin validación con usuarios finales
- Entrenamiento en CPU (lento)

## 12. Próximos Pasos

1. Recolectar datos propios de la empresa con condiciones reales de campo
2. Validar con equipo técnico y agronómico
3. Implementar API de inferencia con FastAPI
4. Crear dashboard para visualización de alertas
5. Establecer pipeline de reentrenamiento con datos validados
