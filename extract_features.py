import os
import csv
import numpy as np
from PIL import Image

BASE = os.path.expanduser(
    "~/Downloads/archive/PlantVillageDataset/train_val_test"
)

OUTPUT = os.path.expanduser(
    "~/smart_agriculture/plant_features.csv"
)

splits = ["train", "val", "test"]

processed = 0
failed = 0

with open(OUTPUT, "w", newline="") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "Image",
        "Disease",
        "Split",
        "Mean_R",
        "Mean_G",
        "Mean_B",
        "Std_R",
        "Std_G",
        "Std_B"
    ])

    for split in splits:

        split_path = os.path.join(BASE, split)

        for disease in sorted(os.listdir(split_path)):

            disease_path = os.path.join(split_path, disease)

            if not os.path.isdir(disease_path):
                continue

            print(f"Processing {split}: {disease}")

            for filename in sorted(os.listdir(disease_path)):

                image_path = os.path.join(disease_path, filename)

                try:
                    image = Image.open(image_path).convert("RGB")
                    image = image.resize((64, 64))

                    pixels = np.asarray(image, dtype=np.float32)

                    means = pixels.mean(axis=(0, 1))
                    stds = pixels.std(axis=(0, 1))

                    writer.writerow([
                        filename,
                        disease,
                        split,
                        *means,
                        *stds
                    ])

                    processed += 1

                except Exception as error:
                    failed += 1
                    print("Failed:", image_path, error)

print("\nFeature extraction completed!")
print("Images processed:", processed)
print("Failed images:", failed)
print("Output:", OUTPUT)
