import csv
import numpy as np

INPUT = "plant_features.csv"
OUTPUT = "plant_vectors_current.tsv"
MAPPING = "plant_mapping_current.csv"

feature_columns = [
    "Mean_R", "Mean_G", "Mean_B",
    "Std_R", "Std_G", "Std_B"
]

with open(INPUT, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract numerical features
X = np.array([
    [float(row[col]) for col in feature_columns]
    for row in rows
], dtype=np.float64)

# Z-score standardization
means = X.mean(axis=0)
stds = X.std(axis=0)

stds[stds == 0] = 1

X_scaled = (X - means) / stds

# Write Mahout-compatible vectors and label mapping
with open(OUTPUT, "w") as vector_file, \
     open(MAPPING, "w", newline="") as mapping_file:

    writer = csv.writer(mapping_file)

    writer.writerow([
        "Key", "Image", "Disease", "Split"
    ])

    for i, row in enumerate(rows):

        key = f"image_{i:06d}"

        vector_file.write(
            key + "\t" +
            "\t".join(f"{value:.8f}" for value in X_scaled[i]) +
            "\n"
        )

        writer.writerow([
            key,
            row["Image"],
            row["Disease"],
            row["Split"]
        ])

print("Standardization completed!")
print("Total vectors:", len(rows))
print("Features per vector:", len(feature_columns))
print("Vector file:", OUTPUT)
print("Mapping file:", MAPPING)
