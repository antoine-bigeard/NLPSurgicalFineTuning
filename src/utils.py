import yaml
import datasets
import logging
import transformers
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict


logging.basicConfig()
LOG = logging.getLogger(__name__)

datasets.logging.set_verbosity_error()


def read_yaml_config_file(path_config: str):
    with open(path_config) as conf:
        return yaml.load(conf, yaml.FullLoader)


def model2hfname(model: str) -> str:
    return {
        "bert-tiny": "prajjwal1/bert-tiny",
        "bert-med": "prajjwal1/bert-medium",
    }[model]


def dataset2hfname(dataset: str) -> str:
    return {
        "mnli": ("multi_nli",),
        "amazon_video": ("amazon_us_reviews", "Video_v1_00"),
        "amazon_books": ("amazon_us_reviews", "Books_v1_00"),
        "amazon_electronics": ("amazon_us_reviews", "Mobile_Electronics_v1_00"),
        "amazon_kitchen": ("amazon_us_reviews", "Kitchen_v1_00"),
        "amazon_shoes": ("amazon_us_reviews", "Shoes_v1_00"),
        "amazon_grocery": ("amazon_us_reviews", "Grocery_v1_00"),
        "amazon_luggage": ("amazon_us_reviews", "Luggage_v1_00"),
        

        "tweet_eval": ("tweet_eval", "offensive"),
        "civil_comments": ("civil_comments",),
    }[dataset]


def stop_tokens(tokenizer, stop_string: str = ".") -> int:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) == stop_string:
            tokens.append(idx)
    return tokens


def get_data(dataset: str, num_samples: int):
    if "amazon" in dataset:
        d = datasets.load_dataset(dataset2hfname(dataset)[0], dataset2hfname(dataset)[1])[
            "train"
        ]
        filter_fn = lambda rows: ["sex" not in r.lower() for r in rows["review_body"]]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        x = d["review_body"]
        y = [s - 1 for s in d["star_rating"]]
        df = defaultdict(lambda: [None] * 5 * num_samples)
        counts = defaultdict(int)
        end_idx = 0
        for idx in range(len(y)):
            c = counts[y[idx]]
            if c < num_samples:
                df["x"][c * 5 + y[idx]] = x[idx]
                df["y"][c * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
                end_idx += 1

        # filter3 = lambda rows: [r is not None and len(r)>0 for r in df["x"]]
        # filter4 = lambda rows: [r is not None and r in [0,1] for r in df["y"]]
        # df = df.filter(filter3, batched=True, batch_size=None)
        # df = df.filter(filter4, batched=True, batch_size=None)

        return df, end_idx
    elif dataset == "tweet_eval":
        d = datasets.load_dataset(dataset2hfname(dataset)[0], dataset2hfname(dataset)[1])[
            "train"
        ]
        filter1 = lambda rows: [r is not None and len(r)>0 for r in rows["text"]]
        filter2 = lambda rows: [r is not None and r in [0,1] for r in rows["label"]]
        # filter_fn = lambda rows: [clean(r, no_emoji=True) for r in rows["text"]]
        d = d.filter(filter1, batched=True, batch_size=None)
        d = d.filter(filter2, batched=True, batch_size=None)
        x = [r for r in d["text"]]
        y = [int(r) for r in d["label"]]

        # print("tweet_eval")
        # print(len(x))
        # print(len(y))

        df = defaultdict(lambda: [None] * 2 * num_samples)
        counts = defaultdict(int)
        end_idx = 0
        for idx in range(len(y)):
            c = counts[y[idx]]
            if c < num_samples:
                df["x"][c * 2 + y[idx]] = x[idx]
                df["y"][c * 2 + y[idx]] = y[idx]
                # print(df["x"][c * 2 + y[idx]], df["y"][c * 2 + y[idx]])
                counts[y[idx]] += 1
                end_idx += 1

        # filter3 = lambda rows: [r is not None and len(r)>0 for r in df["x"]]
        # filter4 = lambda rows: [r is not None and r in [0,1] for r in df["y"]]
        # df = df.filter(filter3, batched=True, batch_size=None)
        # df = df.filter(filter4, batched=True, batch_size=None)
        

        return df, end_idx
    elif dataset == "civil_comments":
        d = datasets.load_dataset(dataset2hfname(dataset)[0])[
            "train"
        ]
        filter1 = lambda rows: [r is not None and len(r)>0 for r in rows["text"]]
        filter2 = lambda rows: [r is not None and r>=0.0 for r in rows["toxicity"]]
        # filter_fn = lambda rows: [clean(r, no_emoji=True) for r in rows["text"]]
        d = d.filter(filter1, batched=True, batch_size=None)
        d = d.filter(filter2, batched=True, batch_size=None)
        # d = d.filter(filter_fn, batched=True, batch_size=None)
        x = [r for r in d["text"]]
        y = [int(0) if r<=0.5 else int(1) for r in d["toxicity"]]

        df = defaultdict(lambda: [None] * 2 * num_samples)
        counts = defaultdict(int)
        end_idx = 0
        for idx in range(len(y)):
            c = counts[y[idx]]
            if c < num_samples:
                df["x"][c * 2 + y[idx]] = x[idx]
                df["y"][c * 2 + y[idx]] = y[idx]
                # print(df["x"][c * 2 + y[idx]], df["y"][c * 2 + y[idx]])
                counts[y[idx]] += 1
                end_idx += 1

        # print("civil_comments")
        # print(len(x))
        # print(len(y))

        # filter3 = lambda rows: [r is not None and len(r)>0 for r in df["x"]]
        # filter4 = lambda rows: [r is not None and r in [0,1] for r in df["y"]]
        # df = df.filter(filter3, batched=True, batch_size=None)
        # df = df.filter(filter4, batched=True, batch_size=None)


        return df, end_idx

    else: ## To be filled with the logic to extract other datasets
        raise NotImplementedError()

"""TO UPDATE FOR NEXT PUSH
"""
# def get_single_dataset_train_val(
#     ds: str,
#     train_pct: List[int],
#     val_pct: List[int],
#     n_train: int,
#     n_val: int = 100,
# ):

#     train_data = defaultdict()
#     val_data = defaultdict()

#     train_samples = int((n_train * train_pct) / 100)
#     val_samples = int((n_val * val_pct) / 100)
#     df_train, _ = get_data(ds, train_samples, mode="train")
#     df_val, _ = get_data(ds, val_samples, mode="val")
#     train_data["x"] = df_train["x"][: 5 * train_samples]
#     train_data["y"] = df_train["y"][: 5 * train_samples]
#     val_data["x"] = df_val["x"][5 * val_samples]
#     val_data["y"] = df_val["y"][5 * val_samples]

#     return train_data, val_data


def get_single_dataset(
    ds: str,
    train_pct: List[int],
    val_pct: List[int],
    n_classes: int,
    n_train: int,
    n_val: int = 100,
):

    train_data = defaultdict()
    val_data = defaultdict()

    train_samples = int((n_train * train_pct) / 100)
    val_samples = int((n_val * val_pct) / 100)
    df, _ = get_data(
        ds,
        train_samples + val_samples,
    )
    train_data["x"] = df["x"][: n_classes * train_samples]
    train_data["y"] = df["y"][: n_classes * train_samples]
    val_data["x"] = df["x"][n_classes * train_samples :]
    val_data["y"] = df["y"][n_classes * train_samples :]

    return train_data, val_data


def get_train_val_pcts(
    train_datasets: List[str],
    val_datasets: List[str],
    train_percentages: List[int],
    val_percentages: List[int],
):
    pcts = {}
    for i, ds in enumerate(train_datasets):
        pcts[ds] = {"train": train_percentages[i], "val": 0}
    for i, ds in enumerate(val_datasets):
        if ds in pcts:
            pcts[ds]["val"] = val_percentages[i]
        else:
            pcts[ds] = {"train": 0, "val": val_percentages[i]}
    return pcts


def get_train_val_datasets(
    train_datasets: List[str],
    val_datasets: List[str],
    train_percentages: List[int],
    val_percentages: List[int],
    n_train: int,
    n_val: int,
):
    train_val_pcts = get_train_val_pcts(
        train_datasets, val_datasets, train_percentages, val_percentages
    )
    train_data = {"x": [], "y": []}
    val_data = {"x": [], "y": []}
    for ds in set(train_datasets + val_datasets):
        n_classes = 5 if "amazon" in ds else 2
        temp_train, temp_val = get_single_dataset(
            ds,
            train_val_pcts[ds]["train"],
            train_val_pcts[ds]["val"],
            n_classes,
            n_train,
            n_val,
        )
        train_data["x"].extend(temp_train["x"])
        train_data["y"].extend(temp_train["y"])
        val_data["x"].extend(temp_val["x"])
        val_data["y"].extend(temp_val["y"])
    
    return train_data, val_data



def get_model_and_tokenizer(model: str, Cls, **model_kwargs):
    hf_model_name = model2hfname(model)

    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({"pad_token": "[PAD]"})
            tok.pad_token = "[PAD]"
    return m, tok


















## GRAVEYARD


# def metric_for_dataset(dataset: str):
#     return {
#         "mnli": "classification accuracy",
#     }[dataset]


# def early_stop_thresold(dataset: str):
#     return {
#         "mnli": 0.95,
#     }[dataset]

# def get_dataset(
#     ds: List[str],
#     train_percentages: List[int],
#     val_percentages: List[int],
#     n_train: int,
#     n_val: int = 100,
# ):

#     train_data = defaultdict()
#     train_data["x"] = []
#     train_data["y"] = []
#     val_data = defaultdict()
#     val_data["x"] = []
#     val_data["y"] = []
#     for i, d in enumerate(ds):
#         dataset = d
#         train_split = train_percentages[i]
#         val_split = val_percentages[i]
#         train_samples = int((n_train * train_split) / 100)
#         val_samples = int((n_val * val_split) / 100)
#         df, _ = get_data(
#             dataset,
#             train_samples,
#         )
#         train_data["x"].extend(df["x"][: 5 * train_samples])
#         train_data["y"].extend(df["y"][: 5 * train_samples])
#         val_data["x"].extend(df["x"][5 * train_samples : 5 * val_samples])
#         val_data["y"].extend(df["y"][5 * train_samples : 5 * val_samples])

#     return train_data, val_data


# def get_dataset(dataset: str, n_train: int, n_val: int = 100):
#     if dataset == "cnn":
#         n_train = 64
#         d = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
#         filter_fn = lambda rows: [
#             "VIDEO" not in a
#             and len(a.split(" ")) < 110
#             and len(a.split(" ")) > 35
#             and len(s.split(" ")) < 25
#             for a, s in zip(rows["article"], rows["highlights"])
#         ]
#         d = d.filter(filter_fn, batched=True, batch_size=None)
#         d = d.rename_columns({"article": "x", "highlights": "y"})

#         def strip_target(row):
#             y = row["y"]
#             y = y.replace(" .", ".")
#             if ". " in y:
#                 y = y[: y.index(". ")]
#             if "\n" in y:
#                 y = y[: y.index("\n")]
#             row["y"] = y
#             return row

#         d = d.map(strip_target)
#         d = d.add_column("simple_y", d["y"])
#         return d[:n_train], d[n_train : n_train + n_val]
#     elif dataset == "trivia":
#         n_train = 256
#         d = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train[:1%]")
#         targets = [
#             [a["normalized_value"]] + a["normalized_aliases"] for a in d["answer"]
#         ]
#         d = d.add_column("simple_y", [t[0] for t in targets])
#         d = d.add_column("y", targets)
#         d = d.rename_column("question", "x")
#         offset = 0
#         return (
#             d[offset : offset + n_train],
#             d[offset + n_train : offset + n_train + n_val],
#         )
#     elif dataset == "babi":
#         n_train = 256
#         d = datasets.load_dataset("babi_qa", "en-valid-10k-qa1", split="train")
#         answer_idxs = []
#         for story in d["story"]:
#             for idx, answer in enumerate(story["answer"]):
#                 if answer:
#                     answer_idxs.append(idx)
#                     break

#         perm = np.random.permutation(len(d["story"]))
#         answers = [story["answer"][idx] for idx, story in zip(answer_idxs, d["story"])]
#         stories = [
#             " ".join(story["text"][: idx + 1])
#             for idx, story in zip(answer_idxs, d["story"])
#         ]

#         answers = [answers[idx] for idx in perm]
#         stories = [stories[idx] for idx in perm]
#         data = {"x": stories, "y": answers, "simple_y": answers}
#         d = datasets.Dataset.from_dict(data)
#         return d[:n_train], d[n_train : n_train + n_val]
#     elif dataset == "amazon":
#         d = datasets.load_dataset("amazon_us_reviews", "Video_v1_00")["train"]
#         filter_fn = lambda rows: ["sex" not in r.lower() for r in rows["review_body"]]
#         d = d.filter(filter_fn, batched=True, batch_size=None)
#         x = d["review_body"]
#         y = [s - 1 for s in d["star_rating"]]
#         train = defaultdict(lambda: [None] * 5 * n_train)
#         val = defaultdict(lambda: [None] * 5 * n_val)
#         counts = defaultdict(int)
#         for idx in range(len(y)):
#             c = counts[y[idx]]
#             if c < n_train:
#                 train["x"][c * 5 + y[idx]] = x[idx]
#                 train["y"][c * 5 + y[idx]] = y[idx]
#                 counts[y[idx]] += 1
#             elif c < n_train + n_val:
#                 val["x"][(c - n_train) * 5 + y[idx]] = x[idx]
#                 val["y"][(c - n_train) * 5 + y[idx]] = y[idx]
#                 counts[y[idx]] += 1
#         return train, val
#     elif dataset == "xsum":
#         n_train = 256
#         d = datasets.load_dataset("xsum", split="train")
#         filter_fn = lambda rows: [
#             len(a.split(" ")) + len(s.split(" ")) < 100
#             for a, s in zip(rows["document"], rows["summary"])
#         ]
#         d = d.filter(filter_fn, batched=True, batch_size=None)
#         d = d.rename_columns({"document": "x", "summary": "y"})
#         d = d.add_column("simple_y", d["y"])
#         return d[:n_train], d[n_train : n_train + n_val]
#     else:
#         raise NotImplementedError(f"{dataset}")




def metric_for_dataset(dataset: str):
    return {
        "mnli": "classification accuracy",
        "amazon_books": "classification accuracy",
        "amazon_video": "classification accuracy",
        "tweet_eval": "classification accuracy",
        "civil_comments": "classification accuracy",
    }[dataset]


def early_stop_thresold(dataset: str):
    return {
        "mnli": 0.95,
        "amazon_books": 0.95,
        "amazon_video": 0.95,
        "tweet_eval": 0.95,
        "civil_comments": 0.95,
    }[dataset]
