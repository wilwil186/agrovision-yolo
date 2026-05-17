# Proyecto de IA con Visión por Computadora y YOLO

## Documento maestro para ejecución asistida con modelo de lenguaje

Este documento consolida dos pruebas técnicas relacionadas con inteligencia artificial aplicada a negocio y visión por computadora. Su objetivo es servir como guía para que un modelo de lenguaje o herramienta de desarrollo asistido pueda entender el contexto completo del proyecto, reconstruir la solución y ejecutar una propuesta técnica usando YOLO.

El documento está dividido en dos partes:

1. **Transcripción estructurada de los documentos originales**: contiene el objetivo, contexto, reto y entregables solicitados.
2. **Guía técnica paso a paso para construir la solución**: describe de forma detallada qué debe hacer el modelo de lenguaje para desarrollar el proyecto correctamente, desde el entendimiento del negocio hasta una implementación inicial con YOLO.

---

# Parte 1. Transcripción estructurada de los documentos originales

## 1. Prueba técnica de conocimientos en IA aplicada a negocio

### 1.1. Objetivo de la prueba

La prueba busca conocer la forma de razonar y proponer soluciones de inteligencia artificial para problemas reales de negocio.

No se busca una única respuesta correcta ni una prueba de memoria. El interés principal está en evaluar cómo la persona candidata:

- Analiza el contexto del problema.
- Formula preguntas relevantes.
- Plantea hipótesis.
- Estructura una solución técnica.
- Lleva una solución de IA a una operación empresarial real.

### 1.2. Indicaciones generales

La persona candidata puede hacer preguntas para aclarar el contexto. La forma en que formule dichas preguntas también hace parte de la evaluación.

Se pueden proponer diferentes tipos de soluciones, entre ellas:

- Machine learning tradicional.
- Redes neuronales.
- Visión por computador.
- Reglas de negocio.
- Modelos híbridos.
- Otras alternativas técnicamente justificadas.

Cada decisión debe ser explicada. En esta fase no es necesario desarrollar código.

La propuesta debe considerar, como mínimo:

- Datos disponibles y datos requeridos.
- Proceso operativo.
- Validación de la solución.
- Métricas de desempeño.
- Despliegue.
- Mantenimiento.
- Impacto en el negocio.

---

## 2. Escenario de negocio: conteo de rosas listas para corte

### 2.1. Contexto del problema

Una empresa del sector floricultor utiliza videos capturados en campo para estimar la cantidad de rosas listas para corte por bloque.

En una evaluación reciente, el sistema proyectó:

- **10.000 rosas listas para corte**, según el modelo.

Sin embargo, el conteo realizado por el jefe de finca fue:

- **8.000 rosas listas para corte**, según la realidad observada en campo.

Esta diferencia genera problemas operativos importantes, especialmente en:

- Planeación logística.
- Asignación de personal.
- Cumplimiento de compromisos comerciales.
- Confianza en las proyecciones automáticas.

### 2.2. Condiciones actuales de captura de datos

Los videos son capturados en condiciones variables de operación. Entre las fuentes de variabilidad se encuentran:

- Diferentes horarios del día.
- Cambios de iluminación.
- Variaciones entre bloques de cultivo.
- Variaciones entre variedades de rosa.
- Presencia de poda reciente.
- Recorridos manuales con cámara en movimiento.
- Posibles inconsistencias en el registro de la información.

Estas condiciones pueden afectar el desempeño del sistema de visión por computadora y explicar parte de la diferencia entre la predicción automática y el conteo humano.

### 2.3. Necesidad de negocio

La empresa busca rediseñar o fortalecer su solución para:

- Mejorar la confiabilidad de las proyecciones.
- Reducir el margen de error en el conteo.
- Entender cuándo una predicción es confiable y cuándo no.
- Identificar posibles causas de error, como oclusión por hojas, duplicidad de fotogramas o inconsistencias de captura.
- Integrar la solución en la operación real de campo.

### 2.4. Instrucciones del escenario

Con base en este contexto, la persona candidata debe:

1. Plantear un enfoque para abordar el problema.
2. Describir cómo estructuraría una solución técnica viable.
3. Indicar qué información consideraría relevante y por qué.
4. Explicar cómo evaluaría el desempeño de la solución.
5. Proponer cómo integraría esta solución en la operación real.
6. Analizar los principales riesgos, limitaciones y decisiones técnicas.

### 2.5. Pregunta central del caso

La empresa procesa videos de camas de cultivo para proyectar tiempos de cosecha. En un bloque específico, el modelo reporta 10.000 rosas listas para corte, pero el jefe de finca solo cuenta 8.000 en la realidad.

La empresa necesita asegurar proyecciones precisas para reducir riesgos logísticos.

La pregunta principal es:

> ¿Cómo se abordaría el diseño de una solución basada en inteligencia artificial que ayude a auditar y reducir este margen de error, por ejemplo, por oclusión de hojas o duplicidad de fotogramas?

La respuesta debe explicar:

- Razonamiento.
- Hipótesis.
- Información necesaria.
- Posibles enfoques.
- Validación.
- Integración con el proceso operativo.

### 2.6. Entregable esperado

El entregable esperado para esta parte es un documento escrito con las consideraciones del candidato.

No se requiere código.

---

## 3. Prueba técnica práctica: solución de IA para clasificación y priorización de condiciones de producto

### 3.1. Nombre de la prueba

**Exploración técnica de una solución de IA para priorización y clasificación de condiciones de producto.**

### 3.2. Objetivo

La prueba busca conocer la forma de razonar, estructurar y comunicar una solución de inteligencia artificial aplicada a una necesidad de negocio.

No se busca una única respuesta correcta. El objetivo es evaluar cómo la persona candidata:

- Analiza el problema.
- Formula preguntas.
- Define supuestos.
- Prepara datos.
- Propone un enfoque técnico.
- Mide si la solución funciona.
- Comunica resultados y limitaciones.

### 3.3. Contexto del caso práctico

La empresa está explorando una solución basada en inteligencia artificial para apoyar procesos de:

- Calidad.
- Trazabilidad.
- Priorización de revisiones en campo.

Actualmente, algunas decisiones operativas dependen de inspecciones visuales manuales sobre:

- Productos.
- Insumos.
- Material vegetal.

Estas inspecciones manuales pueden verse afectadas por:

- Experiencia del personal.
- Volumen de trabajo.
- Variabilidad de las condiciones observadas.
- Oportunidad con la que se detectan anomalías.

Para esta prueba se entregará un conjunto pequeño de imágenes asociado a diferentes condiciones visibles en hojas de uva, tomando como referencia el dataset público **GRAPE Leaf Diseases** de Kaggle.

Este dataset está orientado a tareas de detección de enfermedades en hojas de uva con formato compatible con modelos tipo YOLO.

### 3.4. Necesidad de la compañía

La compañía desea evaluar si, a partir de información visual, es posible construir una solución inicial que permita:

- Detectar condiciones visibles en hojas de uva.
- Clasificar dichas condiciones.
- Alertar casos que requieran revisión por parte del equipo técnico o agronómico.
- Priorizar revisiones manuales.
- Reducir revisiones innecesarias.
- Generar confianza en los usuarios que tomarán decisiones con base en las alertas del sistema.

---

## 4. Reto técnico práctico

### 4.1. Reto principal

Diseñar una propuesta técnica para construir un modelo basado en YOLO que permita identificar condiciones visibles en hojas de uva y apoyar la toma de decisiones operativas.

La respuesta debe explicar cómo abordar el problema desde:

- Entendimiento del negocio.
- Exploración del dataset.
- Preparación de datos.
- Entrenamiento del modelo.
- Validación de resultados.
- Definición de alertas.
- Limitaciones del dataset.
- Uso en campo.
- Integración en un proceso real de calidad o trazabilidad.

### 4.2. Supuestos

La persona candidata puede hacer supuestos razonables, pero debe indicarlos explícitamente.

También debe mencionar qué información adicional solicitaría a la empresa para mejorar:

- Calidad del modelo.
- Utilidad operativa.
- Interpretabilidad de resultados.
- Integración con la operación real.

### 4.3. Dataset

El dataset de referencia es:

**GRAPE Leaf Diseases**

Se entregará una carpeta con un conjunto reducido de archivos y/o una tabla de referencia con etiquetas o categorías.

El dataset es pequeño y está diseñado para evaluar criterio técnico, no para entrenar modelos pesados.

La persona candidata puede trabajar con el enfoque que considere más apropiado, siempre que explique por qué lo eligió.

### 4.4. Actividad solicitada

Con base en el dataset entregado, se debe proponer y desarrollar una solución inicial que permita apoyar la clasificación o priorización de casos.

La entrega debe responder, como mínimo, las siguientes preguntas:

1. ¿Cómo se entiende el problema de negocio y cuál sería el objetivo técnico propuesto?
2. ¿Qué exploración inicial se haría sobre los datos?
3. ¿Qué preparación, limpieza o transformación se aplicaría?
4. ¿Qué tipo de enfoque o modelo se utilizaría y por qué?
5. ¿Cómo se validaría el resultado?
6. ¿Qué métricas se usarían para saber si la solución es útil?
7. ¿Qué riesgos o limitaciones se identifican?
8. ¿Cómo se llevaría esta solución a un entorno real de la compañía?

---

## 5. Entregables esperados

### 5.1. Notebook o script

Debe contener código o pseudocódigo ejecutable, según el alcance definido para la prueba.

Debe permitir reproducir la exploración, preparación de datos, entrenamiento o prueba de concepto.

### 5.2. README corto

Debe incluir:

- Pasos para ejecutar.
- Librerías usadas.
- Supuestos.
- Estructura de carpetas.
- Instrucciones básicas de uso.

### 5.3. Resumen técnico

Debe explicar:

- Enfoque utilizado.
- Métricas calculadas.
- Resultados obtenidos.
- Limitaciones identificadas.
- Recomendaciones de mejora.

### 5.4. Propuesta de puesta en producción en video

Debe ser una explicación breve de cómo se integraría la solución en un proceso empresarial.

El video debe cubrir:

- Flujo propuesto.
- Usuarios involucrados.
- Criterios de alerta.
- Validación humana.
- Mantenimiento del modelo.
- Valor de negocio esperado.

---

## 6. Lineamientos generales

La persona candidata puede usar Python y las librerías que considere necesarias.

Puede usar modelos simples o modelos preentrenados si lo justifica.

No es obligatorio lograr el mayor desempeño posible. Se valorará más:

- Razonamiento.
- Claridad.
- Capacidad de explicar decisiones.
- Criterio técnico.
- Entendimiento del negocio.

Si el dataset no es suficiente, debe explicarse qué datos adicionales se pedirían y por qué.

Si se decide no entrenar un modelo completo, se puede presentar una prueba de concepto razonada con pasos claros.

### 6.1. Criterio de éxito esperado

Una buena entrega debe permitir entender claramente cómo pasar de un conjunto pequeño de datos a una solución inicial con valor de negocio.

Se espera que la persona candidata explique de manera clara y ordenada:

- Decisiones tomadas.
- Supuestos definidos.
- Métricas utilizadas.
- Limitaciones.
- Riesgos.
- Propuesta de implementación real.

---

# Parte 2. Guía técnica detallada para que el modelo de lenguaje construya el proyecto con YOLO

## 7. Objetivo de esta guía técnica

Esta sección debe ser usada como instrucción para una herramienta de desarrollo asistido por modelo de lenguaje.

La herramienta debe leer este documento, recibir la ruta del dataset y construir una solución inicial basada en YOLO para detectar, clasificar y priorizar condiciones visibles en hojas de uva.

La solución no debe enfocarse únicamente en obtener una métrica alta, sino en construir una prueba técnica clara, reproducible y defendible desde el punto de vista de negocio.

---

## 8. Resultado esperado del proyecto

La herramienta debe generar un proyecto completo con, como mínimo, los siguientes componentes:

```text
proyecto_yolo_uva/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset.yaml
├── notebooks/
│   └── 01_exploracion_yolo_uva.ipynb
├── src/
│   ├── config.py
│   ├── data_audit.py
│   ├── prepare_dataset.py
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── predict.py
│   └── alert_rules.py
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   └── reports/
└── docs/
    ├── resumen_tecnico.md
    ├── supuestos_y_limitaciones.md
    └── propuesta_produccion.md
```

Si el dataset entregado tiene otra estructura, la herramienta debe adaptarse a la estructura real sin inventar archivos, columnas, clases ni carpetas inexistentes.

---

## 9. Principios obligatorios para la herramienta

La herramienta debe cumplir las siguientes reglas:

1. **No inventar columnas, clases, etiquetas ni rutas.**
   - Debe inspeccionar la carpeta entregada.
   - Debe detectar qué archivos existen realmente.
   - Debe identificar si hay imágenes, etiquetas YOLO, tablas CSV, YAML o archivos auxiliares.

2. **No asumir estructura YOLO sin verificarla.**
   - Debe validar si existen carpetas como `images/`, `labels/`, `train/`, `valid/`, `test/`.
   - Debe revisar si existe un archivo `.yaml` de configuración del dataset.
   - Debe revisar si las etiquetas están en formato YOLO: `class_id x_center y_center width height`.

3. **Separar clasificación, detección y priorización.**
   - Si hay bounding boxes, el enfoque principal puede ser detección con YOLO.
   - Si solo hay etiquetas por imagen, debe tratarse como clasificación.
   - Si hay severidad, confianza o reglas operativas, se puede construir priorización.

4. **Construir una solución simple, clara y reproducible.**
   - Dataset pequeño implica evitar entrenamiento pesado.
   - Se debe priorizar transferencia de aprendizaje con modelo preentrenado.
   - Se debe documentar cada decisión.

5. **Explicar limitaciones.**
   - Dataset pequeño.
   - Posible sesgo por condiciones controladas.
   - Diferencia entre hojas de uva del dataset y operación real de campo.
   - Variabilidad de iluminación, cámara, fondo, distancia y oclusión.

6. **Diseñar criterios de alerta útiles para negocio.**
   - No basta con detectar una clase.
   - Se debe definir cómo la predicción se transforma en una alerta operativa.

---

## 10. Paso 1: entender el problema de negocio

La herramienta debe comenzar creando una sección de entendimiento del negocio en el README y en el resumen técnico.

Debe explicar que el problema no es únicamente entrenar un modelo de visión por computadora. El problema real es apoyar decisiones operativas relacionadas con calidad, trazabilidad y priorización de revisiones en campo.

### 10.1. Traducción del problema de negocio a problema técnico

La herramienta debe formular el objetivo técnico así:

> Construir una prueba de concepto con YOLO que permita detectar o clasificar condiciones visibles en hojas de uva, generar una predicción con nivel de confianza y convertir esa predicción en una recomendación operativa de revisión o priorización.

### 10.2. Usuarios de la solución

Debe identificar posibles usuarios:

- Equipo técnico o agronómico.
- Personal de calidad.
- Supervisores de campo.
- Equipo de trazabilidad.
- Analistas responsables de monitorear alertas.

### 10.3. Decisiones que la solución debería apoyar

La solución debería ayudar a decidir:

- Qué casos deben revisarse primero.
- Qué imágenes o lotes tienen condiciones visibles anómalas.
- Qué predicciones son confiables.
- Qué predicciones requieren validación humana.
- Qué zonas o lotes podrían requerir seguimiento.

---

## 11. Paso 2: inspeccionar la estructura real del dataset

Antes de escribir código de entrenamiento, la herramienta debe inspeccionar la ruta entregada por el usuario.

Debe listar:

- Número total de archivos.
- Número de imágenes.
- Extensiones encontradas.
- Carpetas existentes.
- Archivos de etiquetas.
- Archivos YAML.
- Archivos CSV o TXT.
- Posibles particiones `train`, `valid`, `test`.

### 11.1. Código esperado para inspección inicial

La herramienta debe crear un script similar a:

```python
from pathlib import Path
from collections import Counter

DATASET_PATH = Path("RUTA_DEL_DATASET")

all_files = [p for p in DATASET_PATH.rglob("*") if p.is_file()]
extensions = Counter(p.suffix.lower() for p in all_files)
folders = sorted({str(p.parent.relative_to(DATASET_PATH)) for p in all_files})

print("Total files:", len(all_files))
print("Extensions:", extensions)
print("Folders:")
for folder in folders:
    print("-", folder)
```

### 11.2. Resultado esperado de este paso

La herramienta debe producir una sección llamada **Auditoría inicial del dataset** con:

- Tabla de extensiones.
- Tabla de carpetas.
- Conteo de imágenes.
- Conteo de etiquetas.
- Detección de posibles inconsistencias.

---

## 12. Paso 3: validar si el dataset está en formato YOLO

La herramienta debe verificar si las etiquetas cumplen el formato YOLO.

Un archivo de etiqueta YOLO normalmente contiene una línea por objeto detectado:

```text
class_id x_center y_center width height
```

Donde:

- `class_id` es el identificador numérico de la clase.
- `x_center` es el centro horizontal normalizado entre 0 y 1.
- `y_center` es el centro vertical normalizado entre 0 y 1.
- `width` es el ancho normalizado entre 0 y 1.
- `height` es el alto normalizado entre 0 y 1.

### 12.1. Validaciones obligatorias

La herramienta debe validar:

- Que cada imagen tenga etiqueta si se trata de detección supervisada.
- Que cada etiqueta tenga imagen correspondiente.
- Que cada línea tenga cinco valores.
- Que `class_id` sea entero.
- Que las coordenadas estén entre 0 y 1.
- Que `width` y `height` sean mayores que 0.
- Que no existan archivos de etiquetas vacíos, salvo que se definan explícitamente como imágenes sin objeto.
- Que las clases declaradas en `dataset.yaml` coincidan con las clases observadas en las etiquetas.

### 12.2. Código esperado para validar etiquetas

```python
from pathlib import Path

labels_dir = Path("RUTA_LABELS")
errors = []
class_ids = set()

for label_file in labels_dir.rglob("*.txt"):
    lines = label_file.read_text().strip().splitlines()
    if not lines:
        errors.append((str(label_file), "empty_label_file"))
        continue

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append((str(label_file), f"line_{line_number}_invalid_number_of_values"))
            continue

        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append((str(label_file), f"line_{line_number}_invalid_type"))
            continue

        if any(x < 0 or x > 1 for x in coords):
            errors.append((str(label_file), f"line_{line_number}_coords_out_of_range"))

        if coords[2] <= 0 or coords[3] <= 0:
            errors.append((str(label_file), f"line_{line_number}_invalid_box_size"))

        class_ids.add(class_id)

print("Classes found:", sorted(class_ids))
print("Errors found:", len(errors))
for error in errors[:20]:
    print(error)
```

---

## 13. Paso 4: hacer exploración visual y estadística

La herramienta debe crear una exploración mínima pero suficiente.

### 13.1. Exploración estadística

Debe calcular:

- Número de imágenes por partición.
- Número de etiquetas por partición.
- Número de objetos por clase.
- Distribución de clases.
- Número promedio de objetos por imagen.
- Imágenes sin anotaciones.
- Tamaño de las imágenes.
- Proporción de bounding boxes por clase.
- Posibles clases desbalanceadas.

### 13.2. Exploración visual

Debe generar ejemplos visuales de imágenes con sus bounding boxes.

Debe guardar muestras en:

```text
outputs/reports/sample_annotations/
```

Cada muestra debe permitir verificar visualmente si:

- La caja cubre correctamente la zona afectada.
- La clase parece coherente.
- Hay objetos sin anotar.
- Hay cajas duplicadas.
- Hay cajas demasiado grandes o pequeñas.
- Hay oclusión.
- Hay ruido visual relevante.

### 13.3. Preguntas que debe responder la exploración

La exploración debe responder:

- ¿El dataset parece suficiente para entrenar un modelo?
- ¿Hay clases con muy pocos ejemplos?
- ¿Las anotaciones parecen consistentes?
- ¿El problema parece más cercano a clasificación o detección?
- ¿Hay evidencia de sesgo por fondo, iluminación o tipo de imagen?
- ¿Es viable entrenar YOLO directamente o conviene solo hacer una prueba de concepto?

---

## 14. Paso 5: preparar el dataset

La herramienta debe preparar los datos según la estructura real encontrada.

### 14.1. Si el dataset ya viene en formato YOLO

Debe:

1. Validar estructura.
2. Crear o corregir `dataset.yaml` solo si la información existe.
3. Confirmar rutas de `train`, `val` y `test`.
4. No modificar etiquetas sin registrar el cambio.
5. Crear una copia procesada en `data/processed/`.

### 14.2. Si el dataset no viene particionado

Debe crear particiones reproducibles:

- `train`: 70 %
- `val`: 20 %
- `test`: 10 %

En datasets muy pequeños puede usar:

- `train`: 80 %
- `val`: 20 %
- Sin `test`, explicando la limitación.

Debe usar semilla fija, por ejemplo:

```python
RANDOM_SEED = 42
```

### 14.3. Si el dataset solo tiene clasificación por imagen

Debe no forzar detección.

En ese caso debe elegir una de estas opciones:

- Usar YOLO en modo clasificación si la librería lo permite.
- Convertir el problema a clasificación con un modelo simple preentrenado.
- Explicar por qué no se puede entrenar detección sin bounding boxes.
- Proponer reetiquetado con bounding boxes como siguiente paso.

### 14.4. Transformaciones recomendadas

Para dataset pequeño, aplicar aumentación moderada:

- Flip horizontal si es agronómicamente válido.
- Cambios leves de brillo.
- Cambios leves de contraste.
- Pequeñas rotaciones.
- Escalado moderado.

No aplicar transformaciones que destruyan la señal visual de enfermedad o condición de la hoja.

---

## 15. Paso 6: seleccionar enfoque y modelo

La herramienta debe justificar el uso de YOLO.

### 15.1. Enfoque recomendado

Usar transferencia de aprendizaje con un modelo YOLO preentrenado, por ejemplo:

- `yolov8n.pt` o modelo nano equivalente.
- `yolov8s.pt` si el dataset y el entorno lo permiten.

El modelo pequeño es preferible porque:

- El dataset es reducido.
- Entrena más rápido.
- Reduce riesgo de sobreajuste.
- Es más fácil de desplegar en ambientes de campo o edge.

### 15.2. Justificación técnica

YOLO es adecuado si existen bounding boxes porque permite:

- Detectar regiones afectadas en la imagen.
- Clasificar la condición visible.
- Generar confianza por detección.
- Procesar imágenes de forma rápida.
- Integrarse en flujos operativos con alertas.

### 15.3. Alternativas si YOLO no es viable

Si el dataset no tiene etiquetas compatibles con detección, la herramienta debe documentar alternativas:

- Clasificación de imagen completa.
- Segmentación si se requieren áreas precisas.
- Modelo híbrido: clasificación inicial + validación humana.
- Recolección adicional de datos y anotación supervisada.

---

## 16. Paso 7: entrenamiento inicial

La herramienta debe entrenar un modelo inicial, salvo que el dataset sea insuficiente o inválido.

### 16.1. Parámetros iniciales recomendados

Para prueba de concepto:

```python
model = YOLO("yolov8n.pt")

model.train(
    data="data/processed/dataset.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    patience=10,
    seed=42,
    project="outputs/training",
    name="yolo_grape_leaf_poc"
)
```

Estos valores deben ajustarse según recursos disponibles.

### 16.2. Consideraciones de entrenamiento

La herramienta debe explicar:

- Por qué usa modelo preentrenado.
- Por qué usa pocas épocas.
- Cómo evita sobreajuste.
- Qué limitaciones tiene entrenar con pocos datos.
- Por qué no debe optimizar hiperparámetros agresivamente en una prueba pequeña.

### 16.3. Control de reproducibilidad

Debe fijar:

- Semilla aleatoria.
- Versión de librerías.
- Ruta exacta del dataset.
- Configuración del entrenamiento.
- Fecha de ejecución.

---

## 17. Paso 8: evaluación técnica

La herramienta debe evaluar el modelo con métricas técnicas y traducirlas a negocio.

### 17.1. Métricas técnicas para detección

Debe reportar, si están disponibles:

- Precision.
- Recall.
- mAP50.
- mAP50-95.
- Matriz de confusión por clase.
- Falsos positivos.
- Falsos negativos.
- Confianza promedio por clase.

### 17.2. Interpretación de métricas

La herramienta debe explicar las métricas de forma operativa:

- **Precision alta**: cuando el sistema alerta, suele tener razón.
- **Recall alto**: el sistema deja pasar pocos casos relevantes sin detectar.
- **Falsos positivos**: generan revisiones innecesarias.
- **Falsos negativos**: pueden dejar pasar condiciones importantes sin atención.
- **mAP**: mide calidad general de detección y localización.

### 17.3. Métricas orientadas a negocio

Debe proponer métricas adicionales:

- Porcentaje de casos priorizados correctamente.
- Reducción estimada de revisiones manuales innecesarias.
- Tiempo promedio de revisión antes y después.
- Tasa de alertas útiles según validación del equipo técnico.
- Nivel de confianza mínimo para automatizar una alerta.
- Porcentaje de predicciones enviadas a revisión humana.

### 17.4. Evaluación por umbrales

La herramienta debe evaluar diferentes umbrales de confianza, por ejemplo:

```text
0.25, 0.40, 0.50, 0.70
```

Para cada umbral debe analizar:

- Cuántas alertas se generan.
- Cuántas detecciones se pierden.
- Cuántos falsos positivos aparecen.
- Qué umbral parece más razonable para una prueba inicial.

---

## 18. Paso 9: definición de reglas de alerta

El modelo no debe entregar solamente predicciones crudas. Debe convertir las predicciones en recomendaciones operativas.

### 18.1. Ejemplo de reglas iniciales

La herramienta puede implementar reglas como:

```python
def assign_priority(predictions):
    if not predictions:
        return "sin_alerta"

    max_confidence = max(p["confidence"] for p in predictions)
    detected_classes = {p["class_name"] for p in predictions}

    if max_confidence >= 0.70:
        return "revision_prioritaria"

    if max_confidence >= 0.40:
        return "revision_manual"

    return "baja_confianza"
```

### 18.2. Reglas recomendadas para negocio

La herramienta debe proponer un esquema de salida como:

| Nivel | Criterio inicial | Acción operativa |
|---|---|---|
| Alta prioridad | Detección de condición relevante con confianza alta | Revisión técnica prioritaria |
| Revisión manual | Confianza media o múltiples detecciones débiles | Validación humana |
| Monitoreo | Confianza baja pero condición posible | Guardar evidencia y revisar por muestreo |
| Sin alerta | Sin detecciones relevantes | No priorizar |

### 18.3. Salida esperada por imagen

El sistema debe generar un archivo CSV o JSON con columnas reales derivadas de la inferencia, por ejemplo:

```text
image_path
predicted_class
confidence
bbox_x1
bbox_y1
bbox_x2
bbox_y2
priority_level
requires_human_review
```

Estas columnas corresponden a la salida del sistema y no a columnas supuestas del dataset original.

---

## 19. Paso 10: análisis de errores

La herramienta debe producir un análisis de errores.

### 19.1. Tipos de error esperados

Debe considerar:

- Falsos positivos por manchas, sombras o ruido visual.
- Falsos negativos por oclusión de hojas.
- Confusión entre enfermedades visualmente similares.
- Errores por imágenes borrosas.
- Errores por iluminación extrema.
- Errores por fondos no representados en entrenamiento.
- Sobreajuste por dataset pequeño.

### 19.2. Evidencias visuales

Debe guardar ejemplos de:

- Predicciones correctas.
- Falsos positivos.
- Falsos negativos.
- Casos de baja confianza.
- Casos ambiguos.

Carpeta sugerida:

```text
outputs/reports/error_analysis/
```

### 19.3. Preguntas para el negocio después del análisis

La herramienta debe dejar formuladas preguntas como:

- ¿Qué errores son más costosos: falsos positivos o falsos negativos?
- ¿Qué condiciones requieren atención inmediata?
- ¿Qué nivel de confianza mínimo acepta el equipo técnico?
- ¿El sistema debe alertar por imagen, planta, cama, lote o bloque?
- ¿Cómo se validará una alerta en campo?

---

## 20. Paso 11: documentación de supuestos y limitaciones

La herramienta debe crear un archivo:

```text
docs/supuestos_y_limitaciones.md
```

### 20.1. Supuestos mínimos a documentar

Debe incluir supuestos como:

- El dataset entregado representa una muestra inicial, no toda la variabilidad real de campo.
- Las clases del dataset corresponden a condiciones visibles relevantes para el negocio.
- Las etiquetas entregadas son confiables, aunque se auditan parcialmente.
- Las imágenes de hojas de uva son una aproximación al tipo de problema visual que la empresa quiere resolver.
- El modelo será usado como apoyo a la priorización, no como reemplazo total del criterio técnico.

### 20.2. Limitaciones mínimas a documentar

Debe incluir:

- Dataset pequeño.
- Posible desbalance de clases.
- Riesgo de sobreajuste.
- Diferencia entre dataset público y condiciones reales de operación.
- Falta de validación con usuarios finales.
- Falta de medición del impacto económico real.
- Necesidad de reentrenamiento con datos propios de la empresa.

---

## 21. Paso 12: propuesta de integración en operación real

La herramienta debe crear un documento:

```text
docs/propuesta_produccion.md
```

### 21.1. Flujo operativo propuesto

Debe proponer un flujo como este:

1. Captura de imagen en campo.
2. Carga automática o manual al sistema.
3. Inferencia con modelo YOLO.
4. Generación de predicciones con confianza.
5. Aplicación de reglas de alerta.
6. Visualización de imagen con bounding boxes.
7. Priorización de revisión técnica.
8. Validación humana.
9. Registro del resultado real.
10. Retroalimentación para mejorar el modelo.

### 21.2. Arquitectura inicial sugerida

Para una prueba de concepto:

```text
Imágenes de campo
      ↓
Carpeta de entrada o formulario web
      ↓
Servicio de inferencia YOLO
      ↓
Motor de reglas de alerta
      ↓
Dashboard o reporte operativo
      ↓
Validación del equipo técnico
      ↓
Base histórica para reentrenamiento
```

### 21.3. Componentes técnicos sugeridos

La herramienta puede proponer:

- API simple con FastAPI.
- Batch de inferencia para carpetas de imágenes.
- Dashboard en Streamlit o herramienta corporativa.
- Almacenamiento de resultados en CSV, base de datos o lakehouse.
- Registro de imágenes, predicciones, confianza y validación humana.

### 21.4. Gobierno y mantenimiento

Debe incluir:

- Monitoreo de drift visual.
- Revisión periódica de métricas.
- Muestreo de predicciones para auditoría humana.
- Reentrenamiento con datos validados.
- Control de versiones de modelo y dataset.
- Trazabilidad entre imagen, predicción y decisión tomada.

---

## 22. Paso 13: conexión con el caso de conteo de rosas

Aunque la parte práctica trabaja con hojas de uva, la herramienta debe explicar cómo el enfoque se relaciona con el caso de conteo de rosas.

### 22.1. Similitudes

Ambos problemas son de visión por computadora aplicada a operación agrícola.

Comparten retos como:

- Variabilidad de iluminación.
- Oclusión.
- Movimiento de cámara.
- Diferencias entre condiciones controladas y campo real.
- Necesidad de validación humana.
- Riesgo de sobreconteo o subdetección.
- Conversión de predicciones en decisiones operativas.

### 22.2. Diferencias

El caso de rosas requiere conteo y posiblemente seguimiento entre fotogramas.

El caso de hojas de uva requiere clasificación o detección de condiciones visibles.

### 22.3. Aprendizajes transferibles

Del proyecto YOLO de hojas de uva se puede transferir al caso de rosas:

- Auditoría de dataset.
- Validación de etiquetas.
- Detección de objetos.
- Métricas de precisión y recall.
- Análisis de falsos positivos.
- Reglas de confianza.
- Revisión humana en casos ambiguos.
- Monitoreo de error por condiciones de captura.

### 22.4. Consideraciones específicas para conteo de rosas

Para el caso de rosas, la herramienta debe recomendar:

- Detectar botones o rosas listas para corte con bounding boxes.
- Evitar duplicidad entre fotogramas usando tracking.
- Asociar detecciones a segmentos de recorrido.
- Controlar velocidad de cámara y solape de frames.
- Estimar conteo por cama, bloque y fecha.
- Medir error absoluto y error porcentual frente a conteo humano.
- Generar intervalos de confianza o banderas de baja confiabilidad.

### 22.5. Métricas sugeridas para conteo de rosas

Además de métricas de detección, se deben usar:

```text
error_absoluto = abs(conteo_predicho - conteo_real)
error_porcentual = abs(conteo_predicho - conteo_real) / conteo_real
sesgo = conteo_predicho - conteo_real
```

Para el ejemplo:

```text
conteo_predicho = 10000
conteo_real = 8000
error_absoluto = 2000
error_porcentual = 25%
sesgo = +2000
```

Esto indica sobreestimación del 25 %.

---

## 23. Paso 14: README que debe generar la herramienta

La herramienta debe crear un `README.md` claro y breve.

### 23.1. Estructura sugerida del README

```markdown
# Solución inicial YOLO para detección de condiciones visibles en hojas de uva

## Objetivo
Construir una prueba de concepto para detectar o clasificar condiciones visibles en hojas de uva y priorizar revisión técnica.

## Estructura del proyecto
Describir carpetas y archivos principales.

## Dataset
Describir la ruta usada, estructura encontrada, clases y limitaciones.

## Instalación
Indicar cómo instalar dependencias.

## Ejecución
Indicar comandos para auditoría, preparación, entrenamiento, evaluación e inferencia.

## Métricas
Explicar métricas técnicas y métricas de negocio.

## Reglas de alerta
Explicar cómo se transforma una predicción en una acción operativa.

## Supuestos
Listar supuestos.

## Limitaciones
Listar limitaciones.

## Próximos pasos
Proponer mejoras y camino a producción.
```

---

## 24. Paso 15: comandos esperados

La herramienta debe dejar comandos claros como estos, ajustados a las rutas reales:

```bash
python src/data_audit.py --dataset_path data/raw
python src/prepare_dataset.py --dataset_path data/raw --output_path data/processed
python src/train_yolo.py --data_yaml data/processed/dataset.yaml
python src/evaluate_yolo.py --model_path outputs/training/yolo_grape_leaf_poc/weights/best.pt --data_yaml data/processed/dataset.yaml
python src/predict.py --model_path outputs/training/yolo_grape_leaf_poc/weights/best.pt --input_path data/processed/test/images --output_path outputs/predictions
```

Si algún comando no puede ejecutarse por falta de datos o estructura, la herramienta debe explicarlo en el README y proponer el paso mínimo necesario para habilitarlo.

---

## 25. Paso 16: reporte técnico final

La herramienta debe generar:

```text
docs/resumen_tecnico.md
```

### 25.1. Contenido mínimo del resumen técnico

Debe incluir:

1. Problema de negocio.
2. Objetivo técnico.
3. Dataset recibido.
4. Auditoría de datos.
5. Preparación realizada.
6. Modelo elegido.
7. Entrenamiento ejecutado o justificación si no se entrenó.
8. Métricas obtenidas.
9. Interpretación de resultados.
10. Reglas de alerta.
11. Riesgos y limitaciones.
12. Recomendaciones para producción.
13. Próximos pasos.

### 25.2. Estilo del resumen

Debe ser técnico, claro y defendible ante negocio.

No debe vender el modelo como solución terminada. Debe presentarlo como prueba de concepto o solución inicial.

---

## 26. Paso 17: criterios de calidad de la entrega

La entrega será buena si permite responder estas preguntas:

- ¿Qué problema de negocio se está resolviendo?
- ¿Por qué YOLO es una opción razonable?
- ¿Qué datos se recibieron realmente?
- ¿Qué calidad tienen las etiquetas?
- ¿Qué preparación se aplicó?
- ¿Qué modelo se entrenó o se propuso?
- ¿Qué métricas se usaron?
- ¿Qué errores se observaron?
- ¿Qué decisiones operativas permite tomar?
- ¿Qué limitaciones tiene?
- ¿Cómo se integraría en campo?
- ¿Qué se necesita para convertirlo en una solución productiva?

---

## 27. Instrucción final para la herramienta de desarrollo asistido

Cuando recibas la ruta del dataset, debes ejecutar el proyecto siguiendo este orden:

1. Leer este documento completo.
2. Inspeccionar la ruta real del dataset.
3. No asumir estructura ni columnas inexistentes.
4. Auditar imágenes, etiquetas, clases y archivos auxiliares.
5. Determinar si el problema es detección, clasificación o priorización.
6. Preparar el dataset de forma reproducible.
7. Crear o validar `dataset.yaml` si aplica.
8. Entrenar un YOLO pequeño con transferencia de aprendizaje si los datos lo permiten.
9. Evaluar con métricas técnicas.
10. Traducir resultados a métricas y acciones de negocio.
11. Generar reglas de alerta.
12. Guardar predicciones y evidencias visuales.
13. Documentar supuestos, limitaciones y riesgos.
14. Proponer integración en operación real.
15. Generar README, resumen técnico y propuesta de producción.

La prioridad no es maximizar una métrica aislada. La prioridad es entregar una solución inicial clara, trazable, reproducible y útil para tomar decisiones operativas.

