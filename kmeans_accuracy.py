import pandas as pd
from scipy.optimize import linear_sum_assignment

# Load existing cluster-disease report
df = pd.read_csv("final_cluster_disease_report.csv")

# Create cluster vs disease contingency table
table = df.pivot_table(
    index="Cluster_ID",
    columns="Disease",
    values="Image_Count",
    aggfunc="sum",
    fill_value=0
)

total_images = table.values.sum()

# Method 1: Majority-mapping accuracy
majority_correct = table.max(axis=1).sum()
majority_accuracy = majority_correct / total_images

# Method 2: Hungarian matching accuracy
row_indices, col_indices = linear_sum_assignment(-table.values)

matched_correct = table.values[row_indices, col_indices].sum()
hungarian_accuracy = matched_correct / total_images

print("\n========== K-MEANS CLUSTERING ACCURACY ==========")
print("Total images:", total_images)
print("Number of clusters:", len(table.index))
print("Number of disease categories:", len(table.columns))

print("\nMajority-mapping accuracy:")
print(f"{majority_accuracy:.4f} ({majority_accuracy * 100:.2f}%)")

print("\nHungarian-matching accuracy:")
print(f"{hungarian_accuracy:.4f} ({hungarian_accuracy * 100:.2f}%)")

print("\nOptimal cluster-to-disease mapping:")
for r, c in zip(row_indices, col_indices):
    count = table.iloc[r, c]
    print(f"Cluster {table.index[r]} -> {table.columns[c]}: {count} images")

print("\n================================================")
