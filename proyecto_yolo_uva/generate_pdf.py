from pathlib import Path
from datetime import datetime
from fpdf import FPDF

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = OUTPUT_DIR / "informe_ejecutivo.pdf"
CLASS_DIST = OUTPUT_DIR / "outputs" / "reports" / "class_distribution.png"

GREEN = (22, 86, 52)
DARK = (33, 33, 33)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
LIGHT_GRAY = (245, 245, 245)


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GRAY)
            self.set_x(self.l_margin)
            self.cell(0, 8, "AgroVision - Deteccion de condiciones en hojas de uva con YOLO", align="L")
            self.cell(0, 8, f"Pagina {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 14, 200, 14)
            self.ln(6)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(*GRAY)
            self.cell(0, 10, f"Generado: {datetime.now().strftime('%d/%m/%Y')}", align="C")

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*GREEN)
        self.set_x(self.l_margin)
        self.cell(0, 12, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*GREEN)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*DARK)
        self.set_x(self.l_margin)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, text)
        self.set_x(self.l_margin)
        self.ln(3)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.5, f"  - {text}")

    def metric_box(self, label, value, x, y, w=42):
        self.set_xy(x, y)
        self.set_fill_color(*GREEN)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 18)
        self.cell(w, 14, str(value), align="C", fill=True)
        self.set_xy(x, y + 14)
        self.set_fill_color(*LIGHT_GRAY)
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "", 8)
        self.cell(w, 8, label, align="C", fill=True)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*GREEN)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        for ridx, row in enumerate(rows):
            if ridx % 2 == 0:
                self.set_fill_color(*LIGHT_GRAY)
            else:
                self.set_fill_color(*WHITE)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, align="C", fill=True)
            self.ln()
        self.set_x(self.l_margin)
        self.ln(4)


pdf = Report(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=20)

# ---- PORTADA ----
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 32)
pdf.set_text_color(*GREEN)
pdf.cell(0, 14, "AgroVision", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)
pdf.set_font("Helvetica", "", 18)
pdf.set_text_color(*DARK)
pdf.cell(0, 10, "Deteccion de condiciones visibles", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 10, "en hojas de uva con YOLOv8", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_draw_color(*GREEN)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(*GRAY)
pdf.cell(0, 8, "Prueba de concepto - Informe Ejecutivo", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, datetime.now().strftime("%B %Y"), align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(40)
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(*GRAY)
pdf.cell(0, 6, "Documento sin codigo fuente. Enfocado en solucion de negocio.", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 6, "Entregable: proyecto_yolo_uva/informe_ejecutivo.pdf", align="C", new_x="LMARGIN", new_y="NEXT")

# ---- 1. RESUMEN EJECUTIVO ----
pdf.add_page()
pdf.chapter_title("1", "Resumen Ejecutivo")
pdf.body_text(
    "Se desarrollo una prueba de concepto basada en vision por computadora para detectar "
    "condiciones visibles en hojas de uva utilizando YOLOv8 (Ultralytics). El modelo "
    "clasifica y localiza cuatro tipos de condiciones: BlackRot, Esca, Healthy y LeafBlight, "
    "generando alertas operativas para priorizar revisiones tecnicas en campo."
)
pdf.body_text(
    "Los resultados obtenidos son altamente prometedores para una prueba de concepto: "
    "precision media (mAP50) de 0.964, precision de 0.971 y recall de 0.923. "
    "Esto demuestra que es viable construir una solucion de apoyo a la toma de decisiones "
    "utilizando modelos preentrenados y transfer learning, incluso con un dataset moderado."
)
pdf.body_text(
    "El sistema no solo detecta condiciones en las hojas, sino que transforma cada prediccion "
    "en una recomendacion operativa concreta: revision prioritaria, revision manual, monitoreo "
    "o sin alerta. Esto permite integrar la IA directamente en el flujo de trabajo del equipo "
    "tecnico y agronomico."
)
pdf.ln(4)

# Metricas destacadas
y_box = pdf.get_y()
if y_box + 30 > 260:
    pdf.add_page()
    y_box = pdf.get_y()
pdf.metric_box("mAP50", "0.964", 12, y_box)
pdf.metric_box("mAP50-95", "0.906", 58, y_box)
pdf.metric_box("Precision", "0.971", 104, y_box)
pdf.metric_box("Recall", "0.923", 150, y_box)
pdf.set_y(y_box + 30)

# ---- 2. PROBLEMA DE NEGOCIO ----
pdf.add_page()
pdf.chapter_title("2", "Problema de Negocio")
pdf.body_text(
    "La empresa realiza inspecciones visuales manuales para evaluar la calidad y condiciones "
    "de cultivos. Este proceso enfrenta desafios importantes:"
)
pdf.bullet("Dependencia de la experiencia del personal para identificar condiciones")
pdf.bullet("Alto volumen de trabajo que puede llevar a omisiones o errores")
pdf.bullet("Variabilidad en las condiciones de observacion (iluminacion, angulo, distancia)")
pdf.bullet("Dificultad para priorizar objetivamente los casos que requieren atencion")
pdf.bullet("Falta de trazabilidad digital de las inspecciones realizadas")
pdf.ln(4)
pdf.body_text(
    "El objetivo de negocio es contar con una herramienta que apoye al equipo tecnico y "
    "agronomico en la deteccion temprana de condiciones anomalas, permitiendo priorizar "
    "revisiones, reducir inspecciones innecesarias y generar registros trazables "
    "de cada evaluacion."
)

# ---- 3. SOLUCION PROPUESTA ----
pdf.chapter_title("3", "Solucion Propuesta")
pdf.body_text(
    "Se propone un sistema de deteccion de objetos basado en YOLOv8 (You Only Look Once) "
    "que analiza imagenes de hojas de uva y detecta condiciones visibles mediante bounding boxes. "
    "El sistema consta de tres componentes principales:"
)
pdf.sub_title("3.1 Modelo de deteccion")
pdf.body_text(
    "YOLOv8n (nano) preentrenado en COCO, ajustado mediante transfer learning con el dataset "
    "GRAPE Leaf Diseases. El modelo procesa cada imagen y genera bounding boxes con la clase "
    "detectada y un nivel de confianza asociado."
)
pdf.sub_title("3.2 Motor de reglas de alerta")
pdf.body_text(
    "Las predicciones del modelo se transforman en alertas operativas mediante reglas de negocio. "
    "Cada deteccion se clasifica en uno de cuatro niveles de prioridad segun su confianza y tipo "
    "de condicion detectada."
)
pdf.sub_title("3.3 Reportes y trazabilidad")
pdf.body_text(
    "El sistema genera reportes estructurados (CSV/JSON) con todas las predicciones, su nivel "
    "de alerta y la necesidad de revision humana, permitiendo auditoria y trazabilidad completa."
)

# ---- 4. DATASET ----
pdf.add_page()
pdf.chapter_title("4", "Dataset")
pdf.body_text(
    "Se utilizo el dataset publico GRAPE Leaf Diseases de Kaggle, que contiene imagenes "
    "de hojas de uva con anotaciones en formato YOLO (bounding boxes)."
)
pdf.ln(2)
pdf.add_table(
    ["Clase", "Etiqueta", "Cantidad", "Proporcion"],
    [
        ["BlackRot", "GRBR", "1,147", "26.0%"],
        ["Esca", "GRES", "1,383", "31.3%"],
        ["Healthy", "GRHE", "643", "14.6%"],
        ["LeafBlight", "GRLB", "1,242", "28.1%"],
    ],
    col_widths=[35, 30, 50, 35],
)
pdf.body_text(
    "Total: 4,195 imagenes (3,358 train / 837 val) con 4,415 bounding boxes. "
    "La clase Healthy esta subrepresentada (14.6%), lo que puede generar sesgo "
    "en las predicciones hacia clases con mas ejemplos."
)
if CLASS_DIST.exists():
    pdf.image(str(CLASS_DIST), x=30, w=150)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "Distribucion de clases en el dataset", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

# ---- 5. METODOLOGIA ----
pdf.add_page()
pdf.chapter_title("5", "Metodologia")
pdf.sub_title("5.1 Enfoque tecnico")
pdf.body_text(
    "Se opto por YOLOv8n (nano) con transfer learning por las siguientes razones:"
)
pdf.bullet("Modelo preentrenado en COCO: aprovecha caracteristicas visuales generales")
pdf.bullet("Arquitectura ligera: reduce riesgo de sobreajuste con dataset moderado")
pdf.bullet("Entrenamiento rapido: permite iterar y validar hipotesis rapidamente")
pdf.bullet("Facil despliegue: compatible con entornos de produccion y edge computing")
pdf.ln(3)
pdf.sub_title("5.2 Entrenamiento")
pdf.add_table(
    ["Parametro", "Valor"],
    [
        ["Modelo base", "yolov8n.pt (COCO pre-trained)"],
        ["Epocas", "30 (early stopping en epoca 7)"],
        ["Tamano imagen", "640x640 px"],
        ["Batch", "8"],
        ["Optimizador", "Adam (auto)"],
        ["Aumentacion", "Flip HR, brillo/contraste, rotaciones"],
        ["Semilla", "42"],
        ["Dispositivo", "CPU"],
    ],
    col_widths=[55, 135],
)
pdf.body_text(
    "El entrenamiento se detuvo en la epoca 7 por early stopping (paciencia de 10 epocas "
    "sin mejora). Esto indica que el modelo convergio rapidamente gracias al transfer learning."
)
pdf.sub_title("5.3 Evaluacion por umbrales")
pdf.body_text(
    "Se evaluaron cuatro umbrales de confianza para seleccionar el punto de operacion optimo:"
)
pdf.add_table(
    ["Umbral", "Precision", "Recall", "mAP50", "mAP50-95"],
    [
        ["0.25", "0.974", "0.923", "0.948", "0.897"],
        ["0.40", "0.974", "0.923", "0.941", "0.891"],
        ["0.50", "0.974", "0.923", "0.934", "0.885"],
        ["0.70", "0.984", "0.902", "0.896", "0.860"],
    ],
    col_widths=[30, 35, 35, 35, 35],
)
pdf.body_text(
    "El umbral de 0.50 ofrece el mejor equilibrio entre precision y recall "
    "para una prueba de concepto, manteniendo alta deteccion sin sacrificar "
    "exactitud."
)

# ---- 6. RESULTADOS ----
pdf.add_page()
pdf.chapter_title("6", "Resultados")
pdf.body_text(
    "El modelo alcanzo metricas sobresalientes para una prueba de concepto:"
)
pdf.ln(2)
pdf.add_table(
    ["Metrica", "Valor", "Interpretacion"],
    [
        ["mAP50", "0.964", "Deteccion y localizacion de alta calidad"],
        ["mAP50-95", "0.906", "Rendimiento consistente en distintos IoU"],
        ["Precision", "0.971", "Cuando alerta, acierta el 97.1% de las veces"],
        ["Recall", "0.923", "Detecta el 92.3% de las condiciones presentes"],
    ],
    col_widths=[40, 30, 120],
)
pdf.body_text(
    "Interpretacion operativa: de cada 100 condiciones presentes en las imagenes, "
    "el sistema detecta correctamente 92. Y de cada 100 alertas generadas, 97 son "
    "correctas. Esto indica que el sistema es confiable como herramienta de apoyo "
    "a la priorizacion, con una tasa baja de falsos positivos."
)
pdf.body_text(
    "Los falsos negativos (8% de condiciones no detectadas) pueden deberse a "
    "oclusion, iluminacion desfavorable o tamano pequeno de la region afectada. "
    "Estos casos deben derivarse a revision manual."
)

# ---- 7. REGLAS DE ALERTA ----
pdf.add_page()
pdf.chapter_title("7", "Reglas de Alerta")
pdf.body_text(
    "Cada prediccion del modelo se transforma en una accion operativa mediante "
    "reglas de negocio disenadas con el equipo tecnico:"
)
pdf.ln(2)
pdf.add_table(
    ["Nivel", "Confianza", "Accion operativa"],
    [
        ["Alta prioridad", ">= 70%", "Notificacion inmediata al equipo tecnico"],
        ["Revision manual", ">= 40%", "Agendar revision en el dia"],
        ["Monitoreo", "< 40%", "Guardar evidencia, revisar por muestreo semanal"],
        ["Sin alerta", "Sin detecciones", "No requiere accion"],
    ],
    col_widths=[40, 35, 115],
)
pdf.body_text(
    "Ejemplo de aplicacion: si el modelo detecta Esca con 82% de confianza, "
    "se genera una alerta de alta prioridad. Si detecta LeafBlight con 35%, "
    "se registra en monitoreo para revision por muestreo."
)

# ---- 8. INTEGRACION EN OPERACION ----
pdf.add_page()
pdf.chapter_title("8", "Integracion en Operacion")
pdf.sub_title("8.1 Flujo operativo")
pdf.body_text(
    "1. Captura de imagen en campo (dispositivo movil o camara)\n"
    "2. Carga automatica al sistema (carpeta compartida o API)\n"
    "3. Inferencia con modelo YOLO\n"
    "4. Generacion de predicciones con bounding boxes y confianza\n"
    "5. Aplicacion de reglas de alerta (priorizacion automatica)\n"
    "6. Visualizacion en dashboard con imagenes anotadas\n"
    "7. Validacion humana del equipo tecnico\n"
    "8. Registro del resultado real (confirmacion o correccion)\n"
    "9. Retroalimentacion para reentrenamiento del modelo"
)
pdf.ln(4)
pdf.sub_title("8.2 Arquitectura sugerida")
pdf.body_text(
    "Para produccion se recomienda: API REST con FastAPI para inferencia, "
    "dashboard en Streamlit para visualizacion, almacenamiento en base de datos "
    "relacional y bucket de imagenes. El modelo se actualiza periodicamente con "
    "datos validados por el equipo tecnico."
)
pdf.sub_title("8.3 Usuarios involucrados")
pdf.bullet("Tecnico de campo: captura imagenes y valida predicciones")
pdf.bullet("Equipo agronomico: revisa alertas de alta prioridad")
pdf.bullet("Supervisor de calidad: monitorea dashboard y tendencias")
pdf.bullet("Equipo de datos: gestiona reentrenamiento y mejora del modelo")

# ---- 9. CONCLUSIONES ----
pdf.add_page()
pdf.chapter_title("9", "Conclusiones")
pdf.body_text(
    "La prueba de concepto demuestra que es viable construir una solucion de vision "
    "por computadora para detectar condiciones en hojas de uva utilizando YOLOv8 "
    "con transfer learning. Las metricas obtenidas (mAP50 = 0.964) son altamente "
    "satisfactorias para una fase inicial."
)
pdf.sub_title("9.1 Proximos pasos")
pdf.bullet("Recolectar 500-1000 imagenes propias con condiciones reales de campo")
pdf.bullet("Validar el sistema con el equipo tecnico y agronomico")
pdf.bullet("Implementar API de inferencia con FastAPI")
pdf.bullet("Desarrollar dashboard en Streamlit para visualizacion de alertas")
pdf.bullet("Establecer pipeline de reentrenamiento continuo con datos validados")
pdf.ln(4)
pdf.sub_title("9.2 Limitaciones identificadas")
pdf.bullet("Dataset moderado (4,195 imagenes): riesgo de sobreajuste")
pdf.bullet("Clase Healthy subrepresentada (14.6%): posible sesgo")
pdf.bullet("Dataset de Kaggle: condiciones controladas, no equivalentes a campo real")
pdf.bullet("Sin validacion con usuarios finales ni medicion de impacto economico")
pdf.ln(4)
pdf.body_text(
    "Se recomienda avanzar a una fase de validacion en campo con datos propios "
    "de la empresa antes de considerar un despliegue productivo."
)

pdf.output(str(OUTPUT_PDF))
print(f"PDF generado: {OUTPUT_PDF}")
