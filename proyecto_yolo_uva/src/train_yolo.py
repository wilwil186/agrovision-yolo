import argparse
from pathlib import Path

from ultralytics import YOLO

from config import DATASET_YAML, YOLO_MODEL, IMG_SIZE, EPOCHS, BATCH_SIZE, PATIENCE, RANDOM_SEED, OUTPUTS


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo YOLO")
    parser.add_argument("--data_yaml", type=str, default=str(DATASET_YAML))
    parser.add_argument("--model", type=str, default=YOLO_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        print(f"ERROR: No existe {data_yaml}")
        print("Ejecuta primero: python src/prepare_dataset.py")
        return

    print("=" * 60)
    print("ENTRENAMIENTO YOLO")
    print("=" * 60)
    print(f"\n  Dataset YAML: {data_yaml}")
    print(f"  Modelo base:  {args.model}")
    print(f"  Épocas:       {args.epochs}")
    print(f"  Tamaño img:   {args.imgsz}")
    print(f"  Batch:        {args.batch}")
    print(f"  Patience:     {PATIENCE}")
    print(f"  Seed:         {RANDOM_SEED}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=PATIENCE,
        seed=RANDOM_SEED,
        project=str(OUTPUTS / "training"),
        name="yolo_grape_leaf_poc",
        verbose=True,
    )

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")
    print(f"\n  Modelo guardado en: {OUTPUTS / 'training' / 'yolo_grape_leaf_poc' / 'weights'}")

    metrics_file = OUTPUTS / "metrics" / "training_metrics.txt"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "metrics").mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(str(results))
    print(f"  Métricas guardadas en: {metrics_file}")


if __name__ == "__main__":
    main()
