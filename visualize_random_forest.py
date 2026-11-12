import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Select the first decision tree from the forest
tree = model.estimators_[0]

feature_names = [
    "Mean_R", "Mean_G", "Mean_B",
    "Std_R", "Std_G", "Std_B"
]

# Shorten disease names for readability
class_names = [
    name.replace("___", "_").replace("__", "_")
        .replace("Tomato_", "Tom_")
        .replace("Pepper_bell_", "Pepper_")
        .replace("Potato_", "Pot_")
    for name in model.classes_
]

# Generate tree visualization
plt.figure(figsize=(32, 18))

plot_tree(
    tree,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=3,
    impurity=False,
    proportion=True,
    fontsize=8
)

plt.title(
    "Random Forest - Decision Tree 1\nPlant Disease Classification",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    "random_forest_classification_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "random_forest_classification_tree.pdf",
    bbox_inches="tight"
)

plt.close()

print("Random Forest decision tree generated successfully!")
print("Tree selected: 1 out of", len(model.estimators_))
print("Maximum displayed depth: 3")
print("Generated files:")
print("1. random_forest_classification_tree.png")
print("2. random_forest_classification_tree.pdf")
