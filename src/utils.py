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
        "small": "gpt2",
        "med": "gpt2-medium",
        "large": "gpt2-large",
        "full": "gpt2-xl",
        "gpt2-sm": "gpt2",
        "gpt2-med": "gpt2-medium",
        "gpt2-lg": "gpt2-large",
        "gpt2": "gpt2-xl",
        "neo": "EleutherAI/gpt-neo-2.7B",
    }[model]


def dataset2hfname(dataset: str) -> str:
    return {
        "mnli": ("multi_nli",),
        "amazon_video": ("amazon_us_reviews", "Video_v1_00"),
        "amazon_books": ("amazon_us_reviews", "Books_v1_00"),
        "cnn": ("cnn_dailymail", "3.0.0"),
        "math": ("math_qa",),
        "tos": ("ought/raft", "terms_of_service"),
        "xsum": ("xsum",),
        "babi": ("babi_qa", "en-valid-10k-qa1"),
    }[dataset]


def is_qa_dataset(dataset: str) -> bool:
    return dataset in ["trivia", "babi"]


def stop_tokens(tokenizer, stop_string: str = ".") -> int:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) == stop_string:
            tokens.append(idx)
    return tokens


def max_sampled_tokens_for_dataset(dataset: str) -> int:
    return {
        "cnn": 30,
        "trivia": 12,
        "babi": 6,
        "xsum": 30,
    }[dataset]


def get_data(dataset: str, num_samples: int, start_index: int):
    d = datasets.load_dataset(dataset2hfname(dataset)[0], dataset2hfname(dataset)[1])[
        "train"
    ]
    filter_fn = lambda rows: ["sex" not in r.lower() for r in rows["review_body"]]
    d = d.filter(filter_fn, batched=True, batch_size=None)
    x = d["review_body"]
    y = [s - 1 for s in d["star_rating"]]
    df = defaultdict(lambda: [None] * 5 * num_samples)
    counts = defaultdict(int)
    for idx in range(start_index, len(y)):
        c = counts[y[idx]]
        if c < num_samples:
            df["x"][c * 5 + y[idx]] = x[idx]
            df["y"][c * 5 + y[idx]] = y[idx]
            counts[y[idx]] += 1
    return df


def get_dataset(
    ds: List[str],
    percentages: List[int],
    val_dataset: str,
    n_train: int,
    n_val: int = 100,
):
    val_data = get_data(val_dataset, n_val, 0)

    train_data = defaultdict()
    train_data["x"] = []
    train_data["y"] = []
    for i, d in enumerate(ds):
        dataset = d
        split = percentages[i]
        num_samples = int((n_train * split) / 100)
        df = get_data(dataset, num_samples, 5 * n_val)
        train_data["x"].extend(df["x"])
        train_data["y"].extend(df["y"])

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


def metric_for_dataset(dataset: str):
    return {
        "cnn": "rouge",
        "xsum": "rouge",
        "trivia": "exact match",
        "babi": "exact match",
        "amazon_books": "classification accuracy",
        "amazon_video": "classification accuracy",
    }[dataset]


def early_stop_thresold(dataset: str):
    return {
        "cnn": 0.8,
        "trivia": 0.7,
        "babi": 0.9,
        "amazon": 0.75,
        "xsum": 0.55,
    }[dataset]
