import pathlib

import numpy as np
import IPython
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")


def draw_standard_bar_plot(
    class_labels, values, results_path, max_value=1.0, y_label="mAP@0.5IOU"
):
    # Create output directory.
    results_path.mkdir(parents=True, exist_ok=True)
    fig_path = results_path / (y_label + ".png")

    objects = tuple(class_labels)
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, values, align="center")
    plt.xticks(y_pos, objects)
    plt.xticks(rotation=80)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(str(fig_path))
    plt.clf()


def draw_precision_recall_curve(precisions, recalls, results_path, name="pr_curve"):
    # Create output directory.
    results_path.mkdir(parents=True, exist_ok=True)
    fig_path = results_path / (name + ".png")

    plt.plot(recalls, precisions, "g--", linewidth=2, markersize=12)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(str(fig_path))
    plt.clf()


def draw_comparison_bar_plot(
    class_labels,
    values_one,
    values_two,
    comparisons,
    results_path,
    y_label="mAP@0.5IOU",
):
    # Create output directory.
    results_path.mkdir(parents=True, exist_ok=True)
    fig_path = results_path / ("comp.png")
    # data to plot
    n_groups = len(class_labels)
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(
        index, values_one, bar_width, alpha=opacity, color="b", label=comparisons[0]
    )

    rects2 = plt.bar(
        index + bar_width,
        values_two,
        bar_width,
        alpha=opacity,
        color="g",
        label=comparisons[1],
    )

    plt.ylabel(y_label)
    plt.xticks(index + bar_width, tuple(class_labels))
    plt.xticks(rotation=80)
    plt.ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(str(fig_path))
    plt.clf()


if __name__ == "__main__":
    class_labels = ["a", "b", "c"]
    class_values = [0.5, 0.2, 1.0]
    results_path = pathlib.Path(f"/data/test_bar_plot")
    draw_standard_bar_plot(class_labels, class_values, results_path)

    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.2, 0.5, 0.8])

    draw_precision_recall_curve(x, y, results_path)

    draw_comparison_bar_plot(
        class_labels,
        class_values,
        class_values,
        ["stuff_one", "stuff_two"],
        results_path,
    )
