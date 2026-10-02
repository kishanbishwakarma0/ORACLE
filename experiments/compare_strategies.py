
"""
ORACLE - Active Learning Strategy Comparison

Compares random and uncertainty sampling using
their saved validation metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load experiment results
random_df = pd.read_csv(
    "experiments/random_sampling_metrics.csv"
)

uncertainty_df = pd.read_csv(
    "experiments/uncertainty_sampling_metrics.csv"
)

# Create comparison directory
output_dir = Path("experiments/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot validation accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    random_df["labeled_samples"],
    random_df["validation_accuracy"],
    marker="o",
    linewidth=2,
    label="Random Sampling",
)

plt.plot(
    uncertainty_df["labeled_samples"],
    uncertainty_df["validation_accuracy"],
    marker="s",
    linewidth=2,
    label="Uncertainty Sampling",
)

plt.xlabel("Number of Labeled Samples")
plt.ylabel("Validation Accuracy (%)")
plt.title("ORACLE: Active Learning Strategy Comparison")

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save high-resolution plot
plot_path = output_dir / "strategy_comparison.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Comparison graph saved at: {plot_path}")

# Generate comparison table
comparison = pd.DataFrame({
    "labeled_samples": random_df["labeled_samples"],
    "random_accuracy": random_df["validation_accuracy"],
    "uncertainty_accuracy": uncertainty_df["validation_accuracy"],
})

comparison["difference"] = (
    comparison["uncertainty_accuracy"]
    - comparison["random_accuracy"]
)

comparison.to_csv(
    "experiments/strategy_comparison.csv",
    index=False,
)

print("\n--- Strategy Comparison ---")
print(comparison.to_string(index=False))

print("\nComparison completed successfully!")