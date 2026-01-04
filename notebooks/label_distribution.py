"""Marimo notebook to visualize label distribution in the speech error dataset."""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import yaml
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from collections import Counter
    import sys

    sys.path.insert(0, "/home/coder/speech-model/src")
    from speech_model.data_cleaning import clean_substitution_error

    sns.set_theme(style="whitegrid")
    return Counter, Path, clean_substitution_error, mo, np, pd, plt, sns, yaml


@app.cell
def _(mo):
    mo.md("""
    # Speech Error Pattern Label Distribution

    This notebook analyzes the distribution of error pattern labels in the speech dataset
    to understand class imbalance and inform model training decisions.
    """)
    return


@app.cell
def _(Path, pd, yaml):
    # Load data
    DATA_DIR = Path("/home/coder/speech-model")
    df = pd.read_parquet(DATA_DIR / "data/processed/utterances.parquet")

    # Load ontology
    with open(DATA_DIR / "ontology.yaml") as f:
        ontology = yaml.safe_load(f)

    error_patterns = sorted(ontology["error_patterns"].keys())
    n_utterances = len(df)
    n_participants = df["participant_id"].nunique()

    print(f"Loaded {n_utterances:,} utterances from {n_participants} participants")
    return df, error_patterns, n_participants, n_utterances


@app.cell
def _(mo, n_participants, n_utterances):
    mo.md(
        f"""
        ## Dataset Overview

        - **Total Utterances**: {n_utterances:,}
        - **Total Participants**: {n_participants}
        - **Utterances per Participant**: {n_utterances / n_participants:.1f} (average)
        """
    )
    return


@app.cell
def _(Counter, df, error_patterns, np, pd):
    # Count occurrences of each error pattern (ORIGINAL)
    label_counts = Counter()
    total_labels = 0

    for error_pattern in df["error_patterns"]:
        if isinstance(error_pattern, (list, np.ndarray)) and len(error_pattern) > 0:
            label_counts.update(error_pattern)
            total_labels += len(error_pattern)

    # Create sorted dataframe for visualization
    count_df = pd.DataFrame(
        [(label, label_counts.get(label, 0)) for label in error_patterns],
        columns=["Error Pattern", "Count"],
    )
    count_df["Percentage"] = (count_df["Count"] / total_labels * 100).round(2)
    count_df["Utterance Coverage %"] = (count_df["Count"] / len(df) * 100).round(2)
    count_df = count_df.sort_values("Count", ascending=False)

    avg_labels_per_utterance = total_labels / len(df)
    return avg_labels_per_utterance, count_df, total_labels


@app.cell
def _(Counter, clean_substitution_error, df, error_patterns, np, pd):
    # Count occurrences with CLEANED labels
    label_counts_cleaned = Counter()
    total_labels_cleaned = 0

    for err_pattern in df["error_patterns"]:
        clnd = clean_substitution_error(err_pattern)
        if isinstance(clnd, (list, np.ndarray)) and len(clnd) > 0:
            label_counts_cleaned.update(clnd)
            total_labels_cleaned += len(clnd)

    # Create sorted dataframe for cleaned labels
    count_df_cleaned = pd.DataFrame(
        [(label, label_counts_cleaned.get(label, 0)) for label in error_patterns],
        columns=["Error Pattern", "Count"],
    )
    count_df_cleaned["Percentage"] = (count_df_cleaned["Count"] / total_labels_cleaned * 100).round(
        2
    )
    count_df_cleaned["Utterance Coverage %"] = (count_df_cleaned["Count"] / len(df) * 100).round(2)
    count_df_cleaned = count_df_cleaned.sort_values("Count", ascending=False)

    avg_labels_per_utterance_cleaned = total_labels_cleaned / len(df)

    # Calculate impact on substitution_error
    original_subst = (
        label_counts_cleaned.get("substitution_error", 0) if "label_counts" in dir() else 0
    )
    cleaned_subst = label_counts_cleaned.get("substitution_error", 0)
    return (
        avg_labels_per_utterance_cleaned,
        count_df_cleaned,
        total_labels_cleaned,
    )


@app.cell
def _(
    avg_labels_per_utterance,
    avg_labels_per_utterance_cleaned,
    mo,
    total_labels,
    total_labels_cleaned,
):
    mo.md(
        f"""
        ## Label Statistics

        ### Original Labels
        - **Total Labels**: {total_labels:,}
        - **Average Labels per Utterance**: {avg_labels_per_utterance:.2f}

        ### Cleaned Labels (substitution_error removed when other patterns exist)
        - **Total Labels**: {total_labels_cleaned:,}
        - **Average Labels per Utterance**: {avg_labels_per_utterance_cleaned:.2f}
        - **Labels Removed**: {total_labels - total_labels_cleaned:,} ({(total_labels - total_labels_cleaned) / total_labels * 100:.1f}%)
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Label Counts Comparison

    Comparing original vs. cleaned label distributions.
    """)
    return


@app.cell
def _(count_df, count_df_cleaned, pd):
    # Create comparison table focusing on top patterns
    comparison_df = pd.DataFrame(
        {
            "Error Pattern": count_df.head(15)["Error Pattern"],
            "Original Count": count_df.head(15)["Count"].values,
            "Cleaned Count": count_df_cleaned.set_index("Error Pattern")
            .loc[count_df.head(15)["Error Pattern"]]["Count"]
            .values,
        }
    )
    comparison_df["Difference"] = comparison_df["Original Count"] - comparison_df["Cleaned Count"]
    comparison_df["% Change"] = (
        (comparison_df["Cleaned Count"] - comparison_df["Original Count"])
        / comparison_df["Original Count"]
        * 100
    ).round(1)

    comparison_df.style.background_gradient(subset=["Difference"], cmap="RdYlGn_r")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Label Counts Table (Original)
    """)
    return


@app.cell
def _(count_df):
    # Display the table
    count_df.style.background_gradient(subset=["Count"], cmap="YlOrRd")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Visualizations

    ### 1. Impact of Label Cleaning
    Comparison of original vs. cleaned label distributions (top 15 patterns).
    """)
    return


@app.cell
def _(count_df, count_df_cleaned, pd, plt, sns):
    # Create comparison visualization for top 15 patterns
    _top_n_comparison = 15
    _top_patterns_comparison = count_df.head(_top_n_comparison)["Error Pattern"].values

    comparison_data = []
    for pattern in _top_patterns_comparison:
        original = count_df[count_df["Error Pattern"] == pattern]["Count"].values[0]
        cleaned = count_df_cleaned[count_df_cleaned["Error Pattern"] == pattern]["Count"].values[0]
        comparison_data.append({"Pattern": pattern, "Count": original, "Type": "Original"})
        comparison_data.append({"Pattern": pattern, "Count": cleaned, "Type": "Cleaned"})

    comparison_plot_df = pd.DataFrame(comparison_data)

    fig_comp, ax_comp = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=comparison_plot_df,
        y="Pattern",
        x="Count",
        hue="Type",
        ax=ax_comp,
        palette=["#e74c3c", "#2ecc71"],
    )
    ax_comp.set_title(
        "Label Distribution: Original vs. Cleaned (Top 15)", fontsize=14, fontweight="bold"
    )
    ax_comp.set_xlabel("Count", fontsize=12)
    ax_comp.set_ylabel("Error Pattern", fontsize=12)
    ax_comp.legend(title="Dataset")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### 2. Distribution of Label Counts (Original)
    Shows the frequency of each error pattern across all utterances.
    """)
    return


@app.cell
def _(count_df, plt, sns):
    # Create main visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Absolute counts
    ax1 = axes[0]
    sns.barplot(
        data=count_df,
        x="Count",
        y="Error Pattern",
        ax=ax1,
        palette="viridis",
    )
    ax1.set_title("Error Pattern Frequency (Absolute Counts)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Number of Occurrences", fontsize=12)
    ax1.set_ylabel("Error Pattern", fontsize=12)

    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.0f")

    # Plot 2: Percentage of total labels
    ax2 = axes[1]
    sns.barplot(
        data=count_df,
        x="Percentage",
        y="Error Pattern",
        ax=ax2,
        palette="plasma",
    )
    ax2.set_title("Error Pattern Distribution (% of Total Labels)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Percentage of Total Labels (%)", fontsize=12)
    ax2.set_ylabel("Error Pattern", fontsize=12)

    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3. Class Imbalance Analysis (Original)

    Understanding the severity of class imbalance in original data.
    """)
    return


@app.cell
def _(count_df, np, plt):
    # Analyze class imbalance
    counts = count_df["Count"].values
    max_count = counts.max()
    min_count = counts[counts > 0].min() if (counts > 0).any() else 0

    imbalance_ratio = max_count / min_count if min_count > 0 else np.inf

    # Create imbalance visualization
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Log scale visualization
    ax_log = axes2[0]
    y_positions = range(len(count_df))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(count_df)))

    ax_log.barh(y_positions, count_df["Count"], color=colors)
    ax_log.set_yticks(y_positions)
    ax_log.set_yticklabels(count_df["Error Pattern"])
    ax_log.set_xlabel("Count (log scale)", fontsize=12)
    ax_log.set_title("Label Distribution (Log Scale)", fontsize=14, fontweight="bold")
    ax_log.set_xscale("log")
    ax_log.grid(True, alpha=0.3)

    # Cumulative percentage
    ax_cum = axes2[1]
    count_df_sorted = count_df.sort_values("Count", ascending=False).reset_index(drop=True)
    cumulative_pct = count_df_sorted["Count"].cumsum() / count_df_sorted["Count"].sum() * 100

    ax_cum.plot(range(len(cumulative_pct)), cumulative_pct, marker="o", linewidth=2, markersize=6)
    ax_cum.axhline(y=50, color="r", linestyle="--", label="50% of labels")
    ax_cum.axhline(y=80, color="orange", linestyle="--", label="80% of labels")
    ax_cum.set_xlabel("Number of Error Patterns (ranked by frequency)", fontsize=12)
    ax_cum.set_ylabel("Cumulative Percentage of Total Labels (%)", fontsize=12)
    ax_cum.set_title("Cumulative Label Distribution", fontsize=14, fontweight="bold")
    ax_cum.grid(True, alpha=0.3)
    ax_cum.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print(f"Most frequent: {count_df.iloc[0]['Error Pattern']} ({max_count} occurrences)")
    if min_count > 0:
        min_label = count_df[count_df["Count"] == min_count].iloc[0]["Error Pattern"]
        print(f"Least frequent: {min_label} ({min_count} occurrences)")

    # Find how many patterns account for 80% of labels
    n_patterns_80 = (cumulative_pct <= 80).sum() + 1
    print(f"\n{n_patterns_80} patterns account for 80% of all labels")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3. Labels per Utterance Distribution

    How many error patterns appear in each utterance?
    """)
    return


@app.cell
def _(df, np, plt, sns):
    # Count labels per utterance
    labels_per_utterance = df["error_patterns"].apply(
        lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax_hist = axes3[0]
    ax_hist.hist(
        labels_per_utterance,
        bins=range(0, labels_per_utterance.max() + 2),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    ax_hist.set_xlabel("Number of Error Patterns per Utterance", fontsize=12)
    ax_hist.set_ylabel("Number of Utterances", fontsize=12)
    ax_hist.set_title("Distribution of Labels per Utterance", fontsize=14, fontweight="bold")
    ax_hist.grid(True, alpha=0.3)

    # Box plot
    ax_box = axes3[1]
    sns.boxplot(y=labels_per_utterance, ax=ax_box, color="lightcoral")
    ax_box.set_ylabel("Number of Error Patterns", fontsize=12)
    ax_box.set_title("Labels per Utterance (Box Plot)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

    print(f"Labels per utterance statistics:")
    print(f"  Mean: {labels_per_utterance.mean():.2f}")
    print(f"  Median: {labels_per_utterance.median():.0f}")
    print(f"  Min: {labels_per_utterance.min()}")
    print(f"  Max: {labels_per_utterance.max()}")
    print(f"  Std: {labels_per_utterance.std():.2f}")

    # Count utterances with no labels
    no_labels = (labels_per_utterance == 0).sum()
    print(f"\nUtterances with no labels: {no_labels} ({no_labels / len(df) * 100:.1f}%)")
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4. Co-occurrence Analysis

    Which error patterns tend to appear together?
    """)
    return


@app.cell
def _(count_df, df, error_patterns, np, plt, sns):
    # Create co-occurrence matrix
    n_patterns = len(error_patterns)
    cooccurrence = np.zeros((n_patterns, n_patterns))
    pattern_to_idx = {pattern: idx for idx, pattern in enumerate(error_patterns)}

    for error_list in df["error_patterns"]:
        if isinstance(error_list, (list, np.ndarray)) and len(error_list) > 1:
            for i, pattern1 in enumerate(error_list):
                for pattern2 in error_list[i:]:
                    if pattern1 in pattern_to_idx and pattern2 in pattern_to_idx:
                        idx1 = pattern_to_idx[pattern1]
                        idx2 = pattern_to_idx[pattern2]
                        cooccurrence[idx1, idx2] += 1
                        if idx1 != idx2:
                            cooccurrence[idx2, idx1] += 1

    # Plot co-occurrence heatmap (top patterns only for readability)
    _top_n_cooccurrence = 15
    _top_patterns_cooccurrence = count_df.head(_top_n_cooccurrence)["Error Pattern"].values
    _top_indices = [pattern_to_idx[p] for p in _top_patterns_cooccurrence]

    cooccurrence_subset = cooccurrence[np.ix_(_top_indices, _top_indices)]

    fig4, ax4 = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cooccurrence_subset,
        xticklabels=_top_patterns_cooccurrence,
        yticklabels=_top_patterns_cooccurrence,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax4,
        cbar_kws={"label": "Co-occurrence Count"},
    )
    ax4.set_title(
        f"Error Pattern Co-occurrence (Top {_top_n_cooccurrence} Patterns)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Recommendations

    Based on the class imbalance analysis, consider:

    1. **Class Weighting**: Use `pos_weight` in `BCEWithLogitsLoss` to give higher weight to rare classes
    2. **Threshold Tuning**: Optimize prediction thresholds per-class instead of using 0.5 globally
    3. **Focal Loss**: Consider focal loss to focus on hard-to-classify examples
    4. **Oversampling**: Use techniques like SMOTE or duplication for rare classes
    5. **Stratified Splitting**: Ensure all classes are represented in train/validation splits
    6. **Evaluation Metrics**: Focus on per-class metrics rather than just macro-averaged F1
    """)
    return


if __name__ == "__main__":
    app.run()
