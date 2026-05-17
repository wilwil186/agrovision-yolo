import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from config import DATASET_YAML, METRICS_DIR


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

    class_names = ["Grape__BlackRot", "Grape__Esca", "Grape__Healthy", "Grape__LeafBlight"]
    print(f"\n  Métricas globales:")
    print(f"    mAP50:    {results.box.map50:.4f}")
    print(f"    mAP50-95: {results.box.map:.4f}")

    P_mean = float(results.box.p.mean()) if hasattr(results.box.p, 'mean') else float(results.box.p)
    R_mean = float(results.box.r.mean()) if hasattr(results.box.r, 'mean') else float(results.box.r)
    print(f"    Precision: {P_mean:.4f}")
    print(f"    Recall:    {R_mean:.4f}")

    print(f"\n  Métricas por clase:")
    class_data = []
    for cls_idx, name in enumerate(class_names):
        p_cls, r_cls, m50_cls, m95_cls = results.box.class_result(cls_idx)
        p_cls, r_cls, m50_cls, m95_cls = float(p_cls), float(r_cls), float(m50_cls), float(m95_cls)
        class_data.append({"class": name, "precision": round(p_cls, 4), "recall": round(r_cls, 4), "mAP50": round(m50_cls, 4), "mAP50-95": round(m95_cls, 4)})
        print(f"    {name}:")
        print(f"      Precision: {p_cls:.4f}, Recall: {r_cls:.4f}, mAP50: {m50_cls:.4f}, mAP50-95: {m95_cls:.4f}")

    df_classes = pd.DataFrame(class_data)
    df_classes.to_csv(METRICS_DIR / "per_class_metrics.csv", index=False)

    metrics_summary = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": P_mean,
        "recall": R_mean,
    }
    df_summary = pd.DataFrame([metrics_summary])
    df_summary.to_csv(METRICS_DIR / "evaluation_metrics.csv", index=False)
    print(f"\n  Métricas guardadas en: {METRICS_DIR}")

    print(f"\n{'='*60}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
