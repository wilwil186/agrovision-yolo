# Solución inicial YOLO para detección de condiciones visibles en hojas de uva

## Objetivo

Prueba de concepto para detectar condiciones visibles en hojas de uva usando YOLOv8 y convertir las predicciones en alertas operativas para priorizar revisiones técnicas.

## Estructura del proyecto

```
proyecto_yolo_uva/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # Dataset original
│   └── processed/               # Dataset preparado para YOLO
├── notebooks/
│   └── 01_exploracion_yolo_uva.ipynb
├── src/
│   ├── config.py                # Configuración central
│   ├── data_audit.py            # Auditoría del dataset
│   ├── prepare_dataset.py       # Preparación de datos
│   ├── train_yolo.py            # Entrenamiento YOLO
│   ├── evaluate_yolo.py         # Evaluación del modelo
│   ├── predict.py               # Inferencia
│   └── alert_rules.py           # Reglas de alerta para negocio
├── outputs/
│   ├── metrics/                 # Métricas de evaluación
│   ├── predictions/             # Predicciones generadas
│   └── reports/                 # Reportes visuales
└── docs/
    ├── resumen_tecnico.md
    ├── supuestos_y_limitaciones.md
    └── propuesta_produccion.md
```

## Dataset

- **Fuente**: GRAPE Leaf Diseases (Kaggle)
- **Clases**: BlackRot, Esca, Healthy, LeafBlight
- **Particiones**: 3,358 train / 837 val
- **Formato**: YOLO object detection (bounding boxes)

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

### 1. Auditoría del dataset

```bash
python src/data_audit.py
```

### 2. Preparación del dataset

```bash
python src/prepare_dataset.py
```

### 3. Entrenamiento

```bash
python src/train_yolo.py
```

### 4. Evaluación

```bash
python src/evaluate_yolo.py --model_path outputs/training/yolo_grape_leaf_poc/weights/best.pt
```

### 5. Inferencia

```bash
python src/predict.py --model_path outputs/training/yolo_grape_leaf_poc/weights/best.pt --input_path data/processed/images/val
```

### 6. Reglas de alerta

```bash
python src/alert_rules.py --predictions_csv outputs/predictions/predictions.csv
```

## Supuestos

- El dataset representa una muestra inicial de condiciones en hojas de uva
- Las etiquetas son confiables como base para una prueba de concepto
- El modelo apoya la priorización, no reemplaza el criterio técnico

## Limitaciones

- Dataset pequeño (4,195 imágenes) — riesgo de sobreajuste
- Desbalance de clases (Healthy subrepresentada)
- Dataset de Kaggle ≠ condiciones reales de campo
- Sin validación con usuarios finales

## Próximos pasos

1. Recolectar datos propios con condiciones reales de campo (500-1000 imágenes)
2. Validar con equipo técnico y agronómico
3. Implementar API con FastAPI para inferencia
4. Dashboard en Streamlit para visualización de alertas
5. Pipeline de reentrenamiento continuo
