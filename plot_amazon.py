import matplotlib.pyplot as plt
import json
import numpy as np


def plot_curve(
    list_files,
    title,
    abs,
    ord,
    color_codes,
    labels=["(80, 20)", "(60, 40)", "(40, 60)", "(20, 80)", "(0, 100)"],
    color=None,
):
    
    modes = ["all", "first", "middle", "last"]
    for mode in modes:
        values = []
        for file in list_files[mode]:
            with open(file) as f:
                data = json.load(f)
                values.append(data["metric"])
        
        plt.plot(np.array(values), label="fine-tuning " + mode, c=color_codes[mode])
        
        # lab = [item.get_text() for item in ax.get_xticklabels()]
        # if labels is not None:
        #     for i, l in enumerate(labels):
        #         lab[i] = l
        


if __name__ == "__main__":

    color_codes = dict({
        "eval_only" : "royalblue",
        "first": "forestgreen",
        "middle": "gold",
        "last": "darkorange",
        "all": "firebrick",
        "pimped_bert": "darkorchid",
    })
    list_files = dict()


    list_files["all"] = [
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_80-20_all_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_60-40_val_pct_60-40_all_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_40-60_val_pct_40-60_all_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_20-80_val_pct_20-80_all_finetune_and_eval.json",
        "results/ft/bert-small_train_amazon_video_val_amazon_video_train_pct_100_val_pct_100_all_finetune_and_eval.json",
    ]

    list_files["first"] = [
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_80-20_first_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_60-40_val_pct_60-40_first_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_40-60_val_pct_40-60_first_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_20-80_val_pct_20-80_first_finetune_and_eval.json",
        "results/ft/bert-small_train_amazon_video_val_amazon_video_train_pct_100_val_pct_100_first_finetune_and_eval.json",
    ]

    list_files["middle"] = [
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_80-20_middle_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_60-40_val_pct_60-40_middle_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_40-60_val_pct_40-60_middle_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_20-80_val_pct_20-80_middle_finetune_and_eval.json",
        "results/ft/bert-small_train_amazon_video_val_amazon_video_train_pct_100_val_pct_100_middle_finetune_and_eval.json",
    ]

    list_files["last"] = [
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_80-20_val_pct_80-20_last_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_60-40_val_pct_60-40_last_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_40-60_val_pct_40-60_last_finetune_and_eval.json",
        "results/ft/bert-small_amazon_books-amazon_video_val_amazon_books-amazon_video_train_pct_20-80_val_pct_20-80_last_finetune_and_eval.json",
        "results/ft/bert-small_train_amazon_video_val_amazon_video_train_pct_100_val_pct_100_last_finetune_and_eval.json",
    ]

    fig, ax = plt.subplots()
    title = "Evaluation of model trained on Civil Comments on dataset shifts."
    abs = "distribution shift"
    ord = "accuracy (val)"
    # save_path = "results/ft/figure3.jpg"
    labels = ["(80, 20)", "(60, 40)", "(40, 60)", "(20, 80)", "(0, 100)"]

    plot_curve(list_files, title, abs, ord, color_codes, labels)
    plt.xticks([i for i in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_xlabel(abs)
    ax.set_ylabel(ord)
    ax.set_title(title)
    plt.legend()
    # plt.savefig(save_path)