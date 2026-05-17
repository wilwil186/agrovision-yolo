import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from config import DATASET_YAML, OUTPUTS, METRICS_DIR, CONFIDENCE_THRESHOLDS


def evaluate_thresholds(results, thresholds):
    print(f"\n--- Evaluación por umbrales de confianza ---")
    rows = []
    for thresh in thresholds:
        metrics = results if hasattr(results, 'box') else None
        if metrics is None:
            continue
        row = {
            "umbral": thresh,
            "precision": getattr(metrics, 'p', None),
            "recall": getattr(metrics, 'r', None),
            "mAP50": getattr(metrics, 'map50', None),
            "mAP50-95": getattr(metrics, 'map', None),
        }
        rows.append(row)
        print(f"\n  Umbral {thresh:.2f}:")
        print(f"    Precision:  {row['precision']:.4f}" if row['precision'] else "    Precision: N/A")
        print(f"    Recall:     {row['recall']:.4f}" if row['recall'] else "    Recall: N/A")
        print(f"    mAP50:      {row['mAP50']:.4f}" if row['mAP50'] else "    mAP50: N/A")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = METRICS_DIR / "threshold_evaluation.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n  Resultados guardados en: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo YOLO")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_yaml", type=str, default=str(DATASET_YAML))
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: No existe el modelo en {model_path}")
        return

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        print(f"ERROR: No existe {data_yaml}")
        return

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 60)
    print(f"\n  Modelo:  {model_path}")
    print(f"  Data:    {data_yaml}")

    model = YOLO(str(model_path))

    results = model.val(data=str(data_yaml))
    print(f"\n  mAP50:    {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.p:.4f}")
    print(f"  Recall:    {results.box.r:.4f}")

    metrics_summary = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.p,
        "recall": results.box.r,
    }
    df_summary = pd.DataFrame([metrics_summary])
    csv_path = METRICS_DIR / "evaluation_metrics.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  Métricas guardadas en: {csv_path}")

    evaluate_thresholds(results, CONFIDENCE_THRESHOLDS)

    print(f"\n{'='*60}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
