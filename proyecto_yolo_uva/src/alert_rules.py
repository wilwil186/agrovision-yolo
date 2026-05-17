import argparse
import json
from pathlib import Path

import pandas as pd

from config import PREDICTIONS_DIR, ALERT_HIGH_CONF, ALERT_MED_CONF


def assign_priority(predictions):
    if not predictions:
        return "sin_alerta"

    max_confidence = max(p["confidence"] for p in predictions)
    detected_classes = {p["predicted_class"] for p in predictions}

    has_disease = any(c != "Grape__Healthy" for c in detected_classes)

    if has_disease and max_confidence >= ALERT_HIGH_CONF:
        return "revision_prioritaria"

    if has_disease and max_confidence >= ALERT_MED_CONF:
        return "revision_manual"

    if has_disease:
        return "monitoreo"

    return "sin_alerta"


def apply_alerts(predictions_csv: Path):
    if not predictions_csv.exists():
        print(f"ERROR: No existe {predictions_csv}")
        return

    df = pd.read_csv(predictions_csv)
    if df.empty:
        print("No hay predicciones para evaluar.")
        return

    predictions_by_image = df.groupby("image_path").apply(
        lambda g: g.to_dict("records")
    ).to_dict()

    alerts = []
    for image_path, preds in predictions_by_image.items():
        priority = assign_priority(preds)
        requires_review = priority in ("revision_prioritaria", "revision_manual")
        n_detections = len(preds)
        max_conf = max(p["confidence"] for p in preds) if preds else 0
        classes_detected = list(set(p["predicted_class"] for p in preds))

        alerts.append({
            "image_path": image_path,
            "priority_level": priority,
            "requires_human_review": requires_review,
            "num_detections": n_detections,
            "max_confidence": round(max_conf, 4),
            "classes_detected": classes_detected,
        })

    df_alerts = pd.DataFrame(alerts)

    alert_counts = df_alerts["priority_level"].value_counts()
    print(f"\n=== DISTRIBUCIÓN DE ALERTAS ===")
    for level, count in alert_counts.items():
        print(f"  {level}: {count}")

    output_dir = predictions_csv.parent
    alerts_csv = output_dir / "alerts.csv"
    df_alerts.to_csv(alerts_csv, index=False)
    print(f"\n  Alertas guardadas en: {alerts_csv}")

    actionable = df_alerts[df_alerts["requires_human_review"]]
    print(f"\n  Casos que requieren revisión humana: {len(actionable)}")
    if not actionable.empty:
        print(f"\n  --- Muestras de alta prioridad ---")
        high = df_alerts[df_alerts["priority_level"] == "revision_prioritaria"]
        for _, row in high.head(5).iterrows():
            print(f"    {row['image_path']}: conf={row['max_confidence']}, clases={row['classes_detected']}")

    return df_alerts


def main():
    parser = argparse.ArgumentParser(description="Reglas de alerta para negocio")
    parser.add_argument("--predictions_csv", type=str,
                        default=str(PREDICTIONS_DIR / "predictions.csv"))
    args = parser.parse_args()

    csv_path = Path(args.predictions_csv)

    print("=" * 60)
    print("REGLAS DE ALERTA")
    print("=" * 60)
    print(f"\n  Predicciones: {csv_path}")
    print(f"  Umbral alta prioridad: >= {ALERT_HIGH_CONF}")
    print(f"  Umbral revisión manual: >= {ALERT_MED_CONF}")

    apply_alerts(csv_path)

    print(f"\n{'='*60}")
    print("ALERTAS GENERADAS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
