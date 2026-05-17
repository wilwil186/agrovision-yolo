import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from config import (
    CLASS_NAMES, PREDICTIONS_DIR, OUTPUTS, REPORTS_DIR,
    ERROR_ANALYSIS_DIR, CONFIDENCE_THRESHOLDS,
)


def draw_boxes(image_path, predictions, output_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for pred in predictions:
        x1 = pred["bbox_x1"]
        y1 = pred["bbox_y1"]
        x2 = pred["bbox_x2"]
        y2 = pred["bbox_y2"]
        conf = pred["confidence"]
        cls_name = pred["predicted_class"]
        label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser(description="Inferencia con YOLO")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=str(PREDICTIONS_DIR))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: No existe el modelo en {model_path}")
        return

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"ERROR: No existe {input_path}")
        return

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INFERENCIA CON YOLO")
    print("=" * 60)
    print(f"\n  Modelo:   {model_path}")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Confianza mínima: {args.conf}")

    model = YOLO(str(model_path))

    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in image_extensions])
    print(f"\n  Imágenes encontradas: {len(image_files)}")

    if not image_files:
        print("  No hay imágenes para procesar.")
        return

    results = model([str(p) for p in image_files], conf=args.conf)

    all_predictions = []
    for img_path, result in zip(image_files, results):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            all_predictions.append({
                "image_path": str(img_path.name),
                "predicted_class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox_x1": round(x1, 2),
                "bbox_y1": round(y1, 2),
                "bbox_x2": round(x2, 2),
                "bbox_y2": round(y2, 2),
            })

    df = pd.DataFrame(all_predictions)
    csv_path = output_path / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Predicciones guardadas: {csv_path}")
    print(f"  Total detecciones: {len(df)}")

    json_path = output_path / "predictions.json"
    json_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"  JSON guardado: {json_path}")

    annotated_dir = output_path / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    predictions_by_image = {}
    for pred in all_predictions:
        predictions_by_image.setdefault(pred["image_path"], []).append(pred)
    for img_file in image_files:
        if img_file.name in predictions_by_image:
            draw_boxes(img_file, predictions_by_image[img_file.name], annotated_dir)
    print(f"  Imágenes anotadas guardadas en: {annotated_dir}")

    print(f"\n{'='*60}")
    print("INFERENCIA COMPLETADA")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
