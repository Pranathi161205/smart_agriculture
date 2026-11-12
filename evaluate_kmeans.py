import re
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

# ==========================================
# 1. LOAD FEATURE VECTORS
# ==========================================

vectors = pd.read_csv(
    "plant_vectors_current.tsv",
    sep="\t",
    header=None,
    names=["Key", "F1", "F2", "F3", "F4", "F5", "F6"]
)

# ==========================================
# 2. LOAD ACTUAL DISEASE LABELS
# ==========================================

mapping = pd.read_csv("plant_mapping_current.csv")

# ==========================================
# 3. NORMALIZE FEATURE VALUES
# ==========================================

def normalize_features(values):
    result = []

    for value in values:
        rounded = round(float(value), 3)

        # Convert both -0.000 and 0.000 to 0.0
        if rounded == 0:
            rounded = 0.0

        result.append(rounded)

    return tuple(result)


# ==========================================
# 4. EXTRACT CLUSTER ASSIGNMENTS
# ==========================================

cluster_map = {}

with open("clustered_points_dump.txt", "r") as file:

    for line in file:

        match = re.search(
            r"Key:\s*(\d+):.*vec:\s*\[([^\]]+)\]",
            line
        )

        if match:

            cluster_id = int(match.group(1))

            feature_values = normalize_features(
                match.group(2).split(",")
            )

            if feature_values in cluster_map:

                if cluster_map[feature_values] != cluster_id:

                    raise ValueError(
                        "Duplicate feature vector found in different clusters!"
                    )

            else:
                cluster_map[feature_values] = cluster_id


# ==========================================
# 5. MATCH FEATURES WITH CLUSTERS
# ==========================================

def get_feature_key(row):

    values = [
        row[f"F{i}"]
        for i in range(1, 7)
    ]

    return normalize_features(values)


vectors["Feature_Key"] = vectors.apply(
    get_feature_key,
    axis=1
)

vectors["Cluster_ID"] = vectors["Feature_Key"].map(
    cluster_map
)


# ==========================================
# 6. MERGE DISEASE LABELS
# ==========================================

data = vectors.merge(
    mapping[["Key", "Disease"]],
    on="Key",
    how="inner",
    validate="one_to_one"
)


# ==========================================
# 7. VALIDATE MATCHING
# ==========================================

unmatched = data["Cluster_ID"].isna().sum()

print("\n========== DATA VALIDATION ==========")

print("Total vectors:", len(vectors))
print("Total mapped records:", len(data))
print("Matched images:", len(data) - unmatched)
print("Unmatched images:", unmatched)

if len(data) != 20638:
    raise ValueError("Expected 20,638 records. Check the input files.")

if unmatched > 0:
    raise ValueError(
        f"{unmatched} images could not be matched. "
        "Evaluation stopped to prevent incorrect results."
    )

data["Cluster_ID"] = data["Cluster_ID"].astype(int)

print("Number of clusters:", data["Cluster_ID"].nunique())
print("Disease categories:", data["Disease"].nunique())


# ==========================================
# 8. PREPARE EVALUATION DATA
# ==========================================

X = data[
    ["F1", "F2", "F3", "F4", "F5", "F6"]
].values

true_labels = data["Disease"].values

cluster_labels = data["Cluster_ID"].values


# ==========================================
# 9. CALCULATE EVALUATION METRICS
# ==========================================

print("\n========== K-MEANS EVALUATION ==========")

# Silhouette Score (sample of 5000)
silhouette = silhouette_score(
    X,
    cluster_labels,
    sample_size=5000,
    random_state=42
)

# Adjusted Rand Index
ari = adjusted_rand_score(
    true_labels,
    cluster_labels
)

# Normalized Mutual Information
nmi = normalized_mutual_info_score(
    true_labels,
    cluster_labels
)

# Homogeneity
homogeneity = homogeneity_score(
    true_labels,
    cluster_labels
)

# Completeness
completeness = completeness_score(
    true_labels,
    cluster_labels
)

# V-measure
v_measure = v_measure_score(
    true_labels,
    cluster_labels
)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

metrics = {
    "Silhouette Score": silhouette,
    "Adjusted Rand Index (ARI)": ari,
    "Normalized Mutual Information (NMI)": nmi,
    "Homogeneity": homogeneity,
    "Completeness": completeness,
    "V-Measure": v_measure
}

print("\n========== FINAL RESULTS ==========")

for name, value in metrics.items():

    print(f"{name}: {value:.4f}")


# ==========================================
# 11. SAVE RESULTS TO CSV
# ==========================================

results = pd.DataFrame(
    metrics.items(),
    columns=["Metric", "Score"]
)

results.to_csv(
    "kmeans_evaluation_metrics.csv",
    index=False
)

print("\n====================================")
print("K-MEANS EVALUATION COMPLETED!")
print("Total images evaluated:", len(data))
print("Results saved to: kmeans_evaluation_metrics.csv")
print("====================================")
