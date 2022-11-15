# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


# from src.data.datamodule.datamodule import MyDataModule
# from src.model.model.pretrained_model import PretrainedTorchModel
# from src.model.lit_model.lit_model import MyLitModel
# from src.utils import read_yaml_config_file
import sys

sys.path.insert(0, ".")

from src.utils import *

import argparse
import copy
import os
import itertools
import json
import torch
import torch.nn as nn
import transformers
import tqdm
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--dataset")
parser.add_argument("--percentages")
parser.add_argument("--val_dataset")
parser.add_argument("--mode", default="all")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

DEVICE = torch.device(args.device)


def parameters_to_fine_tune(model: nn.Module, mode: str) -> List:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.

    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)

    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """
    # YOUR CODE HERE
    if mode == "all":
        params = [p for p in model.parameters() if p.requires_grad]
        return params
    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    loss = nn.functional.cross_entropy(logits, targets)
    return loss


def get_acc(logits, targets):
    y = torch.argmax(logits, dim=-1) == targets
    y = y.type(torch.float)
    return torch.mean(y).item()


def ft_bert(model, tok, x, y, mode, batch_size=8):
    model = copy.deepcopy(model)

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    all_x = tok(
        x, return_tensors="pt", padding=True, truncation=True, max_length=100
    ).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok(
            [x[i] for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.debug:
            break

        if step % 10 == 0:
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")
            if total_acc > 0.75:
                break
    return model


def run_ft(
    models: List[str],
    datasets: List[str],
    percentages: List[int],
    val_dataset: str,
    modes: List[str],
    n_train: int = 200,
    n_val: int = 40,
):
    results = {}

    train, val = get_dataset(datasets, percentages, val_dataset, n_train, n_val)

    for model_name, mode in itertools.product(models, modes):
        print(f"Fine-tuning {model_name} on and mode={mode}")
        model, tokenizer = get_model_and_tokenizer(
            model_name, transformers.AutoModelForSequenceClassification, num_labels=5
        )

        for repeat in range(args.repeats):
            if repeat > 0:
                print(f"Beginning repeat #{repeat}")
                fine_tuned = ft_bert(model, tokenizer, train["x"], train["y"], mode)
                val_acc = eval(fine_tuned, tokenizer, val)
                results[
                    "_".join(
                        [model_name, "_".join(datasets), "_".join(percentages), mode]
                    )
                ] = val_acc

            print(results)
            question = "ft"
            if not os.path.exists(f"results/{question}"):
                os.makedirs(f"results/{question}")

            for k_, v in results.items():
                with open(f"results/{question}/{k_}.json", "w") as f:
                    json.dump({"metric": v}, f)
            results = {}


if __name__ == "__main__":
    # percentages = [int(k) for k in args.percentages.split(",")]
    percentages = [80, 20]
    # run_ft(args.model.split(","), args.dataset.split(","), percentages, args.val_dataset, args.mode.split(","))
    run_ft(
        ["bert-med"],
        ["amazon_video", "amazon_books"],
        percentages,
        "amazon_video",
        args.mode.split(","),
    )
