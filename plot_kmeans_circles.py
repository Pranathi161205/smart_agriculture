import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# Actual K-Means cluster results
clusters = [
    ("C1", 16495, 6014),
    ("C2", 13045, 4825),
    ("C3", 4338, 3607),
    ("C4", 547, 3475),
    ("C5", 7957, 2717)
]

total = sum(count for _, _, count in clusters)
max_count = max(count for _, _, count in clusters)

fig, ax = plt.subplots(figsize=(15, 7))

positions = np.arange(len(clusters)) * 4

for i, (name, cluster_id, count) in enumerate(clusters):

    # Circle area is proportional to the image count
    radius = 1.45 * np.sqrt(count / max_count)

    circle = Circle(
        (positions[i], 0),
        radius,
        facecolor=f"C{i}",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.75
    )

    ax.add_patch(circle)

    percentage = (count / total) * 100

    ax.text(
        positions[i], 0.25,
        name,
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold"
    )

    ax.text(
        positions[i], -0.2,
        f"ID: {cluster_id}\n{count:,} images\n{percentage:.2f}%",
        ha="center",
        va="center",
        fontsize=10
    )

ax.set_xlim(-2, positions[-1] + 2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")

ax.set_title(
    "K-Means Clustering - Plant Disease Dataset",
    fontsize=17,
    fontweight="bold",
    pad=20
)

ax.axis("off")

plt.tight_layout()

plt.savefig("kmeans_cluster_circles.png", dpi=300, bbox_inches="tight")
plt.savefig("kmeans_cluster_circles.pdf", bbox_inches="tight")

plt.close()

print("K-Means circular visualization generated!")
print("Total clusters:", len(clusters))
print("Total images:", total)
print("Saved:")
print("1. kmeans_cluster_circles.png")
print("2. kmeans_cluster_circles.pdf")
