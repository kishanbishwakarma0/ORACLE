
"""
ORACLE - Experiment Results Visualization

Generates publication-quality plots from multi-seed
active learning experiment results.

Input files:
    experiments/multi_seed_aggregate.csv
    experiments/multi_seed_test_summary.json
    experiments/multi_seed_metrics.csv

Output files:
    experiments/plots/learning_curves.png
    experiments/plots/final_test_accuracy.png
    experiments/plots/annotation_efficiency.png

No model training is performed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

EXPERIMENT_DIR = Path("experiments")
PLOT_DIR = EXPERIMENT_DIR / "plots"

AGGREGATE_FILE = (
    EXPERIMENT_DIR / "multi_seed_aggregate.csv"
)

TEST_SUMMARY_FILE = (
    EXPERIMENT_DIR / "multi_seed_test_summary.json"
)

RAW_METRICS_FILE = (
    EXPERIMENT_DIR / "multi_seed_metrics.csv"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STRATEGY_LABELS = {
    "random": "Random Sampling",
    "uncertainty": "Uncertainty Sampling",
}

# Use a consistent visual style across all figures.
plt.rcParams.update({
    "figure.figsize": (9, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ==================================================
# DATA LOADING AND VALIDATION
# ==================================================

def load_results():
    """Load and validate all experiment result files."""

    required_files = [
        AGGREGATE_FILE,
        TEST_SUMMARY_FILE,
        RAW_METRICS_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required result file not found: {file_path}"
            )

    aggregate_df = pd.read_csv(
        AGGREGATE_FILE
    )

    raw_df = pd.read_csv(
        RAW_METRICS_FILE
    )

    with open(
        TEST_SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        test_data = json.load(file)

    test_df = pd.DataFrame(
        test_data
    )

    # Validate required columns.
    aggregate_columns = {
        "strategy",
        "budget",
        "mean",
        "std",
    }

    raw_columns = {
        "strategy",
        "seed",
        "budget",
        "validation_accuracy",
        "test_accuracy",
    }

    test_columns = {
        "strategy",
        "mean",
        "std",
    }

    if not aggregate_columns.issubset(
        aggregate_df.columns
    ):

        raise ValueError(
            "Aggregate CSV is missing required columns."
        )

    if not raw_columns.issubset(
        raw_df.columns
    ):

        raise ValueError(
            "Raw metrics CSV is missing required columns."
        )

    if not test_columns.issubset(
        test_df.columns
    ):

        raise ValueError(
            "Test summary JSON is missing required fields."
        )

    expected_strategies = {
        "random",
        "uncertainty",
    }

    if set(aggregate_df["strategy"]) != expected_strategies:

        raise ValueError(
            "Unexpected strategies found in aggregate CSV."
        )

    if set(test_df["strategy"]) != expected_strategies:

        raise ValueError(
            "Unexpected strategies found in test summary."
        )

    return (
        aggregate_df,
        test_df,
        raw_df,
    )


# ==================================================
# HELPER: SAVE FIGURE
# ==================================================

def save_figure(fig, filename):
    """Save a figure in high-resolution PNG format."""

    output_path = PLOT_DIR / filename

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    print(
        f"Saved: {output_path}"
    )

    plt.close(fig)


# ==================================================
# GRAPH 1: LEARNING CURVES
# ==================================================

def plot_learning_curves(aggregate_df):
    """
    Plot validation accuracy against annotation budget.

    Shaded region represents plus/minus one standard
    deviation across random seeds.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for strategy in [
        "random",
        "uncertainty",
    ]:

        subset = (
            aggregate_df[
                aggregate_df["strategy"] == strategy
            ]
            .sort_values("budget")
        )

        budgets = subset["budget"].to_numpy(
            dtype=float
        )

        means = subset["mean"].to_numpy(
            dtype=float
        )

        stds = subset["std"].fillna(
            0
        ).to_numpy(
            dtype=float
        )

        label = STRATEGY_LABELS[strategy]

        ax.plot(
            budgets,
            means,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=label,
        )

        ax.fill_between(
            budgets,
            means - stds,
            means + stds,
            alpha=0.18,
        )

    ax.set_title(
        "Active Learning: Validation Learning Curves",
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Annotation Budget (Labeled Samples)"
    )

    ax.set_ylabel(
        "Validation Accuracy (%)"
    )

    ax.set_xticks(
        sorted(
            aggregate_df["budget"].unique()
        )
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "learning_curves.png",
    )


# ==================================================
# GRAPH 2: FINAL TEST ACCURACY
# ==================================================

def plot_final_test_accuracy(test_df):
    """
    Compare final test accuracy using mean and
    standard deviation across random seeds.
    """

    strategies = [
        "random",
        "uncertainty",
    ]

    means = []
    stds = []
    labels = []

    for strategy in strategies:

        row = test_df[
            test_df["strategy"] == strategy
        ].iloc[0]

        means.append(
            float(row["mean"])
        )

        stds.append(
            float(row["std"])
        )

        labels.append(
            STRATEGY_LABELS[strategy]
        )

    x_positions = np.arange(
        len(strategies)
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=7,
        width=0.55,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(
        "Final Test Accuracy Comparison",
        fontweight="bold",
        pad=15,
    )

    ax.set_ylabel(
        "Test Accuracy (%)"
    )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    # Display mean and standard deviation above bars.
    for bar, mean, std in zip(
        bars,
        means,
        stds,
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 1.5,
            f"{mean:.2f}%\n± {std:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    fig.tight_layout()

    save_figure(
        fig,
        "final_test_accuracy.png",
    )


# ==================================================
# GRAPH 3: ANNOTATION EFFICIENCY
# ==================================================

def plot_annotation_efficiency(aggregate_df):
    """
    Show the change in validation accuracy after
    each additional annotation batch.

    The first budget is shown as zero because there
    is no preceding budget for comparison.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for strategy in [
        "random",
        "uncertainty",
    ]:

        subset = (
            aggregate_df[
                aggregate_df["strategy"] == strategy
            ]
            .sort_values("budget")
            .copy()
        )

        subset["accuracy_gain"] = (
            subset["mean"].diff()
        )

        # First budget has no previous measurement.
        subset["accuracy_gain"] = (
            subset["accuracy_gain"].fillna(0)
        )

        budgets = subset["budget"].to_numpy()

        gains = subset[
            "accuracy_gain"
        ].to_numpy()

        ax.plot(
            budgets,
            gains,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=STRATEGY_LABELS[strategy],
        )

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    ax.set_title(
        "Validation Accuracy Gain per Annotation Budget",
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Annotation Budget (Labeled Samples)"
    )

    ax.set_ylabel(
        "Change in Validation Accuracy (Percentage Points)"
    )

    ax.set_xticks(
        sorted(
            aggregate_df["budget"].unique()
        )
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "annotation_efficiency.png",
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("ORACLE - RESULTS VISUALIZATION")
    print("=" * 60)

    print("\nLoading experiment results...")

    aggregate_df, test_df, raw_df = (
        load_results()
    )

    print(
        f"Loaded {len(raw_df)} raw experiment records."
    )

    print(
        f"Loaded {len(aggregate_df)} aggregate records."
    )

    print("\nGenerating visualizations...")

    plot_learning_curves(
        aggregate_df
    )

    plot_final_test_accuracy(
        test_df
    )

    plot_annotation_efficiency(
        aggregate_df
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated figures:")

    for filename in [
        "learning_curves.png",
        "final_test_accuracy.png",
        "annotation_efficiency.png",
    ]:

        print(
            PLOT_DIR / filename
        )

    print(
        "\nAll figures are saved at 300 DPI."
    )


if __name__ == "__main__":
    main()