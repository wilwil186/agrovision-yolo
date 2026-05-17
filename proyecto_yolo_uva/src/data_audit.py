import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

from config import DATA_RAW, REPORTS_DIR, SAMPLE_ANNOTATIONS_DIR


def audit_dataset(dataset_path: Path):
    all_files = [p for p in dataset_path.rglob("*") if p.is_file()]
    extensions = Counter(p.suffix.lower() for p in all_files)
    folders = sorted({str(p.parent.relative_to(dataset_path)) for p in all_files})

    print("=" * 60)
    print("AUDITORÍA INICIAL DEL DATASET")
    print("=" * 60)
    print(f"\nRuta: {dataset_path}")
    print(f"Total archivos: {len(all_files)}")

    print("\n--- Extensiones ---")
    for ext, count in extensions.most_common():
        print(f"  {ext}: {count}")

    print("\n--- Carpetas ---")
    for folder in folders:
        print(f"  {folder}/")

    images = [p for p in all_files if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    labels = [p for p in all_files if p.suffix.lower() == ".txt"]

    print(f"\n--- Conteo ---")
    print(f"  Imágenes: {len(images)}")
    print(f"  Etiquetas: {len(labels)}")

    train_images = sorted((dataset_path / "grape" / "images" / "train").glob("*"))
    val_images = sorted((dataset_path / "grape" / "images" / "val").glob("*"))
    train_labels = sorted((dataset_path / "grape" / "labels" / "train").glob("*"))
    val_labels = sorted((dataset_path / "grape" / "labels" / "val").glob("*"))

    print(f"  train/images: {len(train_images)}")
    print(f"  train/labels: {len(train_labels)}")
    print(f"  val/images:   {len(val_images)}")
    print(f"  val/labels:   {len(val_labels)}")

    train_img_stems = {p.stem for p in train_images}
    train_lbl_stems = {p.stem for p in train_labels}
    val_img_stems = {p.stem for p in val_images}
    val_lbl_stems = {p.stem for p in val_labels}

    print("\n--- Correspondencia imagen-etiqueta ---")
    print(f"  Train: imágenes sin label: {len(train_img_stems - train_lbl_stems)}")
    print(f"  Train: labels sin imagen:  {len(train_lbl_stems - train_img_stems)}")
    print(f"  Val:   imágenes sin label: {len(val_img_stems - val_lbl_stems)}")
    print(f"  Val:   labels sin imagen:  {len(val_lbl_stems - val_img_stems)}")


def validate_labels(labels_dir: Path):
    label_files = sorted(labels_dir.rglob("*.txt"))
    if not label_files:
        print("  No se encontraron archivos de etiquetas.")
        return set(), []

    errors = []
    class_ids = set()
    annotations_per_file = []

    for label_file in label_files:
        content = label_file.read_text().strip()
        if not content:
            errors.append((str(label_file), "empty_label_file"))
            annotations_per_file.append(0)
            continue

        lines = content.splitlines()
        count = 0
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

            if not (0 <= coords[0] <= 1 and 0 <= coords[1] <= 1 and 0 <= coords[2] <= 1 and 0 <= coords[3] <= 1):
                errors.append((str(label_file), f"line_{line_number}_coords_out_of_range"))

            if coords[2] <= 0 or coords[3] <= 0:
                errors.append((str(label_file), f"line_{line_number}_invalid_box_size"))

            class_ids.add(class_id)
            count += 1
        annotations_per_file.append(count)

    return class_ids, errors, annotations_per_file


def compute_statistics(labels_dir: Path):
    class_counter = Counter()
    boxes_per_image = []
    label_files = sorted(labels_dir.rglob("*.txt"))

    for label_file in label_files:
        content = label_file.read_text().strip()
        if not content:
            boxes_per_image.append(0)
            continue
        lines = content.splitlines()
        count = 0
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    class_counter[class_id] += 1
                    count += 1
                except ValueError:
                    pass
        boxes_per_image.append(count)

    return class_counter, boxes_per_image


def main():
    parser = argparse.ArgumentParser(description="Auditar dataset YOLO")
    parser.add_argument("--dataset_path", type=str, default=str(DATA_RAW))
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    audit_dataset(dataset_path)

    labels_train = dataset_path / "grape" / "labels" / "train"
    labels_val = dataset_path / "grape" / "labels" / "val"

    for split_name, labels_dir in [("Train", labels_train), ("Val", labels_val)]:
        print(f"\n{'='*60}")
        print(f"VALIDACIÓN DE ETIQUETAS - {split_name}")
        print(f"{'='*60}")
        if not labels_dir.exists():
            print(f"  Directorio no encontrado: {labels_dir}")
            continue

        class_ids, errors, ann_counts = validate_labels(labels_dir)

        print(f"  Archivos de etiquetas: {len(list(labels_dir.rglob('*.txt')))}")
        print(f"  Clases encontradas: {sorted(class_ids)}")
        print(f"  Errores: {len(errors)}")
        print(f"  Promedio anotaciones/imagen: {sum(ann_counts)/len(ann_counts):.2f}" if ann_counts else "  Sin datos")
        if errors:
            print("\n  Primeros errores:")
            for e in errors[:10]:
                print(f"    {e[0]}: {e[1]}")
        ann_dist = Counter(ann_counts)
        print(f"\n  Distribución anotaciones/imagen:")
        for k in sorted(ann_dist):
            print(f"    {k} anotación(es): {ann_dist[k]} imágenes")

        class_counter, boxes_per_image = compute_statistics(labels_dir)
        print(f"\n  Distribución de clases:")
        df_classes = pd.DataFrame([
            {"class_id": cid, "class_name": ["Grape__BlackRot", "Grape__Esca", "Grape__Healthy", "Grape__LeafBlight"][cid] if cid < 4 else "unknown", "count": cnt}
            for cid, cnt in sorted(class_counter.items())
        ])
        print(df_classes.to_string(index=False))

    print(f"\n{'='*60}")
    print("Auditoría completada.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
