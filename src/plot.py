import matplotlib.pyplot as plt
import json


def plot_curve(
    list_files,
    title,
    abs,
    ord,
    save_path,
    labels=["(80, 20)", "(60, 40)", "(40, 60)", "(20, 80)", "(60, 40)"],
    color=None,
):
    values = []
    for file in list_files:
        with open(file) as f:
            data = json.load(f)
            values.append(data["metric"])
    fig, ax = plt.subplots()
    plt.plot(values) if color is not None else plt.plot(values, color=color)
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.set_xlabel(abs)
    ax.set_ylabel(ord)
    ax.set_title(title)
    plt.savefig(save_path)


if __name__ == "__main__":
    list_files = [
        "/home/abigeard/NLPSurgicalFineTuning/results/ft/bert-small_train_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_80-20_all_eval_only.json",
        "/home/abigeard/NLPSurgicalFineTuning/results/ft/bert-small_train_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_60-40_all_eval_only.json",
        "/home/abigeard/NLPSurgicalFineTuning/results/ft/bert-small_train_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_40-60_all_eval_only.json",
        "/home/abigeard/NLPSurgicalFineTuning/results/ft/bert-small_train_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_20-80_all_eval_only.json",
        "/home/abigeard/NLPSurgicalFineTuning/results/ft/bert-small_train_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_0-100_all_eval_only.json",
    ]
    title = "Evaluation of model trained on books on dataset shifts."
    abs = "distribution shift"
    ord = "accuracy (val)"
    save_path = "/home/abigeard/NLPSurgicalFineTuning/results/ft/figure.jpg"
    labels = ["(80, 20)", "(60, 40)", "(40, 60)", "(20, 80)", "(60, 40)"]

    plot_curve(list_files, title, abs, ord, save_path, labels)
