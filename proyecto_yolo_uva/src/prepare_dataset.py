import argparse
import shutil
from pathlib import Path

from config import DATA_RAW, DATA_PROCESSED, DATASET_YAML


def prepare_yolo_dataset(raw_path: Path, processed_path: Path):
    processed_path.mkdir(parents=True, exist_ok=True)

    grape_source = raw_path / "grape"
    if not grape_source.exists():
        print(f"ERROR: No se encontró la carpeta grape en {raw_path}")
        return

    for split in ["train", "val"]:
        for subdir in ["images", "labels"]:
            src = grape_source / subdir / split
            dst = processed_path / subdir / split
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for item in src.iterdir():
                    shutil.copy2(item, dst / item.name)
                print(f"  Copiados {len(list(src.iterdir()))} archivos a {dst}")
            else:
                print(f"  AVISO: No existe {src}")

    yaml_content = f"""path: {processed_path.resolve()}
train: images/train
val: images/val

nc: 4
names:
  - Grape__BlackRot
  - Grape__Esca
  - Grape__Healthy
  - Grape__LeafBlight
"""
    DATASET_YAML.write_text(yaml_content)
    print(f"\n  Creado dataset.yaml en {DATASET_YAML}")

    print(f"\n  Resumen:")
    for split in ["train", "val"]:
        img_dir = processed_path / "images" / split
        lbl_dir = processed_path / "labels" / split
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"    {split}: {n_imgs} imágenes, {n_lbls} etiquetas")


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset YOLO")
    parser.add_argument("--dataset_path", type=str, default=str(DATA_RAW))
    parser.add_argument("--output_path", type=str, default=str(DATA_PROCESSED))
    args = parser.parse_args()

    raw_path = Path(args.dataset_path)
    out_path = Path(args.output_path)

    print("=" * 60)
    print("PREPARACIÓN DEL DATASET")
    print("=" * 60)
    print(f"\nOrigen:  {raw_path}")
    print(f"Destino: {out_path}")

    prepare_yolo_dataset(raw_path, out_path)

    print(f"\n{'='*60}")
    print("Preparación completada.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
