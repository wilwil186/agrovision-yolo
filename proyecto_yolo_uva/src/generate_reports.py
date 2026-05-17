from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

from config import (
    DATA_RAW, DATA_PROCESSED, METRICS_DIR, REPORTS_DIR,
    SAMPLE_ANNOTATIONS_DIR, ERROR_ANALYSIS_DIR, PREDICTIONS_DIR,
    CLASS_NAMES, CONFIDENCE_THRESHOLDS, RANDOM_SEED,
    YOLO_MODEL, IMG_SIZE, BATCH_SIZE,
)

plt.rcParams["figure.dpi"] = 150

# ── 1. threshold_evaluation.csv ──
def generate_threshold_evaluation():
    print("=" * 60)
    print("EVALUACIÓN POR UMBRALES")
    print("=" * 60)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(Path("outputs/training/yolo_grape_leaf_poc/weights/best.pt").resolve()))
    dataset_yaml = str(DATA_PROCESSED / "dataset.yaml")

    rows = []
    for conf in CONFIDENCE_THRESHOLDS:
        print(f"\n  Evaluando umbral confianza = {conf} ...")
        results = model.val(data=dataset_yaml, conf=conf, verbose=False)
        p_mean = float(results.box.p.mean())
        r_mean = float(results.box.r.mean())
        m50 = float(results.box.map50)
        m95 = float(results.box.map)
        rows.append({"umbral": conf, "precision": round(p_mean, 4), "recall": round(r_mean, 4), "mAP50": round(m50, 4), "mAP50-95": round(m95, 4)})
        print(f"    Precision: {p_mean:.4f}, Recall: {r_mean:.4f}, mAP50: {m50:.4f}, mAP50-95: {m95:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "threshold_evaluation.csv", index=False)
    print(f"\n  Guardado: {METRICS_DIR / 'threshold_evaluation.csv'}")

# ── 2. class_distribution.png ──
def generate_class_distribution():
    print(f"\n{'='*60}")
    print("GRÁFICO DE DISTRIBUCIÓN DE CLASES")
    print(f"{'='*60}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    class_counter = Counter()
    labels_dir = DATA_RAW / "grape" / "labels"
    for split in ["train", "val"]:
        for lbl_file in sorted((labels_dir / split).rglob("*.txt")):
            content = lbl_file.read_text().strip()
            if not content:
                continue
            for line in content.splitlines():
                parts = line.split()
                if len(parts) == 5:
                    try:
                        class_counter[int(parts[0])] += 1
                    except ValueError:
                        pass

    names = CLASS_NAMES[:len(class_counter)]
    counts = [class_counter[i] for i in sorted(class_counter)]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, counts, color=colors[:len(names)], edgecolor="white", linewidth=1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Número de anotaciones", fontsize=12)
    ax.set_title("Distribución de clases en el dataset", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = REPORTS_DIR / "class_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Guardado: {out_path}")

# ── 3. sample_annotations ──
def generate_sample_annotations():
    print(f"\n{'='*60}")
    print("MUESTRAS ANOTADAS")
    print(f"{'='*60}")
    SAMPLE_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(Path("outputs/training/yolo_grape_leaf_poc/weights/best.pt").resolve()))

    val_images = sorted((DATA_PROCESSED / "images" / "val").glob("*"))
    if not val_images:
        val_images = sorted((DATA_RAW / "grape" / "images" / "val").glob("*"))

    rng = np.random.default_rng(RANDOM_SEED)
    selected = rng.choice(val_images, size=min(20, len(val_images)), replace=False)

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        results = model(str(img_path), conf=0.25, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out_path = SAMPLE_ANNOTATIONS_DIR / img_path.name
        cv2.imwrite(str(out_path), img)

    n_saved = len(list(SAMPLE_ANNOTATIONS_DIR.glob("*")))
    print(f"  Generadas {n_saved} muestras anotadas en: {SAMPLE_ANNOTATIONS_DIR}")

# ── 4. error_analysis ──
def generate_error_analysis():
    print(f"\n{'='*60}")
    print("ANÁLISIS DE ERRORES")
    print(f"{'='*60}")
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(Path("outputs/training/yolo_grape_leaf_poc/weights/best.pt").resolve()))

    val_images = sorted((DATA_PROCESSED / "images" / "val").glob("*"))
    if not val_images:
        val_images = sorted((DATA_RAW / "grape" / "images" / "val").glob("*"))

    labels_dir = DATA_PROCESSED / "labels" / "val"
    if not labels_dir.exists():
        labels_dir = DATA_RAW / "grape" / "labels" / "val"

    rng = np.random.default_rng(RANDOM_SEED + 1)
    n_samples = min(50, len(val_images))
    selected = rng.choice(val_images, size=n_samples, replace=False)

    fp_images = []
    fn_images = []
    tp_images = []

    for img_path in selected:
        label_file = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_file.exists():
            content = label_file.read_text().strip()
            for line in content.splitlines():
                parts = line.split()
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        gt_boxes.append((cls_id, cx, cy, bw, bh))
                    except ValueError:
                        pass

        results = model(str(img_path), conf=0.25, verbose=False)
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_boxes.append((cls_id, conf, x1, y1, x2, y2))

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        fn_count = len(gt_boxes) - len(pred_boxes) if len(gt_boxes) > len(pred_boxes) else 0
        fp_count = len(pred_boxes) - len(gt_boxes) if len(pred_boxes) > len(gt_boxes) else 0

        if fp_count > 0 and len(fp_images) < 8:
            for box in pred_boxes:
                if box[0] not in [g[0] for g in gt_boxes]:
                    x1, y1, x2, y2 = map(int, box[2:6])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, f"FP:{CLASS_NAMES[box[0]]} {box[1]:.2f}",
                                (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(img, "FALSO POSITIVO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            fp_images.append(img)

        if fn_count > 0 and len(fn_images) < 8:
            for g in gt_boxes:
                if g[0] not in [p[0] for p in pred_boxes]:
                    cx, cy, bw, bh = g[1:5]
                    x1 = int((cx - bw / 2) * img.shape[1])
                    y1 = int((cy - bh / 2) * img.shape[0])
                    x2 = int((cx + bw / 2) * img.shape[1])
                    y2 = int((cy + bh / 2) * img.shape[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(img, f"FN:{CLASS_NAMES[g[0]]}",
                                (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(img, "FALSO NEGATIVO", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
            fn_images.append(img)

        if len(pred_boxes) > 0 and fp_count == 0 and fn_count == 0 and len(tp_images) < 8:
            for box in pred_boxes:
                x1, y1, x2, y2 = map(int, box[2:6])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{CLASS_NAMES[box[0]]} {box[1]:.2f}",
                            (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(img, "PREDICCIÓN CORRECTA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            tp_images.append(img)

    categories = {
        "correctas": tp_images,
        "falsos_positivos": fp_images,
        "falsos_negativos": fn_images,
    }
    for cat_name, images in categories.items():
        cat_dir = ERROR_ANALYSIS_DIR / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images):
            out_path = cat_dir / f"{cat_name}_{idx+1}.jpg"
            cv2.imwrite(str(out_path), img)
        print(f"  {cat_name}: {len(images)} imágenes guardadas en {cat_dir}")

if __name__ == "__main__":
    generate_threshold_evaluation()
    generate_class_distribution()
    generate_sample_annotations()
    generate_error_analysis()
    print(f"\n{'='*60}")
    print("REPORTES GENERADOS COMPLETAMENTE")
    print(f"{'='*60}")
