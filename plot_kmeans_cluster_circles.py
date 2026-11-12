import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# Load features
df = pd.read_csv(
    "plant_vectors_current.tsv",
    sep="\t",
    header=None,
    names=["Key", "F1", "F2", "F3", "F4", "F5", "F6"]
)

# Read cluster assignments
cluster_ids = []
with open("clustered_points_dump.txt") as f:
    for line in f:
        if line.startswith("Key:"):
            cluster_ids.append(int(line.split(":")[1].strip().split()[0]))

if len(df) != len(cluster_ids):
    raise ValueError("Feature and cluster counts do not match!")

df["Cluster"] = cluster_ids

# PCA: convert six features into 2D
X = df[["F1", "F2", "F3", "F4", "F5", "F6"]]
coords = PCA(n_components=2).fit_transform(X)

df["X"] = coords[:, 0]
df["Y"] = coords[:, 1]

# Cluster labels
labels = {
    16495: "C1",
    13045: "C2",
    4338: "C3",
    547: "C4",
    7957: "C5"
}

fig, ax = plt.subplots(figsize=(12, 9))

for cluster_id, label in labels.items():
    group = df[df["Cluster"] == cluster_id]
    x = group["X"].values
    y = group["Y"].values

    cx, cy = np.mean(x), np.mean(y)

    # Plot image points
    ax.scatter(x, y, s=5, alpha=0.45, label=label)

    # Circular boundary
    radius = max(
        np.sqrt((x - cx)**2 + (y - cy)**2)
    ) * 1.05

    circle = Ellipse(
        (cx, cy),
        width=2 * radius,
        height=2 * radius,
        fill=False,
        linestyle="--",
        linewidth=2
    )
    ax.add_patch(circle)

    ax.text(
        cx, cy,
        f"{label}\n{len(group):,} images",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8)
    )

ax.set_title("2D K-Means Clustering - PlantVillage Dataset")
ax.set_xlabel("X - Principal Component 1")
ax.set_ylabel("Y - Principal Component 2")

ax.legend(title="Clusters")
ax.grid(alpha=0.2)
ax.axhline(0, linewidth=0.7)
ax.axvline(0, linewidth=0.7)

plt.tight_layout()
plt.savefig("kmeans_cluster_circles.png", dpi=300)
plt.savefig("kmeans_cluster_circles.pdf")
plt.show()
