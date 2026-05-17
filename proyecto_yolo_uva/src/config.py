from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATASET_YAML = DATA_PROCESSED / "dataset.yaml"

OUTPUTS = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS / "metrics"
PREDICTIONS_DIR = OUTPUTS / "predictions"
REPORTS_DIR = OUTPUTS / "reports"
SAMPLE_ANNOTATIONS_DIR = REPORTS_DIR / "sample_annotations"
ERROR_ANALYSIS_DIR = REPORTS_DIR / "error_analysis"

DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42

CLASS_NAMES = ["Grape__BlackRot", "Grape__Esca", "Grape__Healthy", "Grape__LeafBlight"]
NUM_CLASSES = len(CLASS_NAMES)

YOLO_MODEL = "yolov8n.pt"
IMG_SIZE = 640
EPOCHS = 30
BATCH_SIZE = 8
PATIENCE = 10

CONFIDENCE_THRESHOLDS = [0.25, 0.40, 0.50, 0.70]

ALERT_HIGH_CONF = 0.70
ALERT_MED_CONF = 0.40
