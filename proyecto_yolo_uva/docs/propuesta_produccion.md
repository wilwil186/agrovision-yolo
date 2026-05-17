# Propuesta de Integración en Operación Real

## Flujo Operativo Propuesto

1. **Captura de imagen en campo** con dispositivo móvil o cámara
2. **Carga automática o manual** al sistema (carpeta compartida, API o formulario web)
3. **Inferencia con modelo YOLO** (servicio batch o API en tiempo real)
4. **Generación de predicciones** con bounding boxes, clase y confianza
5. **Aplicación de reglas de alerta** (priorización automática)
6. **Visualización de imagen anotada** con bounding boxes y nivel de alerta
7. **Priorización de revisión técnica** según nivel de alerta
8. **Validación humana** por parte del equipo técnico o agronómico
9. **Registro del resultado real** (confirmación o corrección)
10. **Retroalimentación** para reentrenamiento del modelo

## Arquitectura Inicial Sugerida

```
Imágenes de campo
      ↓
Carpeta de entrada o formulario web
      ↓
Servicio de inferencia YOLO (FastAPI)
      ↓
Motor de reglas de alerta (src/alert_rules.py)
      ↓
Dashboard o reporte operativo (Streamlit)
      ↓
Validación del equipo técnico
      ↓
Base histórica para reentrenamiento
```

## Componentes Técnicos

### API de Inferencia (FastAPI)
- Endpoint POST `/predict` para inferencia en tiempo real
- Endpoint POST `/predict-batch` para procesamiento por lotes
- Respuesta JSON con predicciones, bounding boxes y nivel de alerta

### Dashboard (Streamlit)
- Visualización de imágenes originales y anotadas
- Tabla de alertas priorizadas
- Filtros por clase, confianza, fecha
- Historial de predicciones y validaciones

### Almacenamiento
- CSV/JSON para resultados de inferencia (fase inicial)
- Base de datos (SQLite → PostgreSQL) para producción
- Almacenamiento de imágenes en bucket S3 o sistema de archivos

## Usuarios Involucrados

- **Técnico de campo**: Captura imágenes y realiza validación inicial
- **Equipo agronómico**: Revisa alertas de alta prioridad y toma decisiones
- **Supervisor de calidad**: Monitorea dashboard y tendencias
- **Equipo de datos**: Gestiona reentrenamiento y mejora del modelo

## Criterios de Alerta

| Nivel | Prioridad | Acción |
|---|---|---|
| Alta prioridad | 1 (inmediata) | Notificación al equipo técnico |
| Revisión manual | 2 (mismo día) | Agenda revisión |
| Monitoreo | 3 (semanal) | Revisión por muestreo |
| Sin alerta | 4 (ninguna) | Sin acción requerida |

## Validación Humana

- El técnico puede confirmar o rechazar cada predicción
- Las predicciones confirmadas alimentan el conjunto de reentrenamiento
- Las predicciones rechazadas se almacenan como falsos positivos para análisis

## Mantenimiento del Modelo

- **Monitoreo de drift**: Comparar distribución de predicciones vs. entrenamiento
- **Reentrenamiento periódico**: Cada 3-6 meses o al alcanzar 500 nuevas imágenes validadas
- **Control de versiones**: Modelo y dataset versionados con etiquetas semánticas
- **Muestreo de auditoría**: 10% de las predicciones auditadas manualmente

## Valor de Negocio Esperado

- Reducción de revisiones manuales innecesarias
- Priorización objetiva de casos críticos
- Trazabilidad digital de inspecciones
- Detección temprana de condiciones anómalas
- Base para análisis histórico y toma de decisiones
