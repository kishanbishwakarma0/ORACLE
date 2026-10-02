
"""
ORACLE - Multi-Seed Learning Curve

Plots mean validation accuracy with
standard deviation bands for each strategy.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load aggregate results
df = pd.read_csv(
    "experiments/multi_seed_aggregate.csv"
)

# Create output directory
output_dir = Path("experiments/plots")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))

for strategy in ["random", "uncertainty"]:

    strategy_df = (
        df[df["strategy"] == strategy]
        .sort_values("labeled_samples")
    )

    x = strategy_df["labeled_samples"]
    mean = strategy_df["mean"]
    std = strategy_df["std"]

    label = (
        "Random Sampling"
        if strategy == "random"
        else "Uncertainty Sampling"
    )

    plt.plot(
        x,
        mean,
        marker="o",
        linewidth=2,
        label=label,
    )

    plt.fill_between(
        x,
        mean - std,
        mean + std,
        alpha=0.2,
    )

plt.xlabel("Number of Labeled Samples")
plt.ylabel("Mean Validation Accuracy (%)")

plt.title(
    "ORACLE: Multi-Seed Active Learning Comparison"
)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = (
    output_dir / "multi_seed_learning_curve.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Graph saved at: {plot_path}")