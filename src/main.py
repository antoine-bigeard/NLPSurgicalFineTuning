import sys

sys.path.insert(0, ".")
from src.pimped_bert import *

from utils import *

import argparse
import copy
import os
import itertools
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
import tqdm
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--dataset")
parser.add_argument("--train_percentages", default="100")
parser.add_argument("--val_percentages", default="100")
parser.add_argument("--train_dataset", default="amazon_electronics")
parser.add_argument("--val_dataset", default="amazon_electronics")
parser.add_argument("--mode", default="all")
parser.add_argument("--path_ckpt", default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--eval_only", default=0, type=int)
parser.add_argument("--n_train", default=1000, type=int)
parser.add_argument("--n_val", default=100, type=int)
args = parser.parse_args()


DEVICE = torch.device(args.device)


def parameters_to_fine_tune(model: nn.Module, mode: str):
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
    if mode == "all":
        params = [p for p in model.parameters() if p.requires_grad]
        return params
    elif mode == "last":
        return list(model.bert.encoder.layer[-2:].parameters())
    elif mode == "first":
        return list(model.bert.encoder.layer[:2].parameters())
    elif mode == "middle":
        n_trans = len(model.bert.encoder.layer)
        return list(
            model.bert.encoder.layer[n_trans // 2 - 1 : n_trans // 2 + 1].parameters()
        )
    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    loss = nn.functional.cross_entropy(logits, targets)
    return loss


def get_acc(logits, targets):
    y = torch.argmax(logits, dim=-1) == targets
    y = y.type(torch.float)
    return torch.mean(y).item()


def eval(model, tok, val_data, batch_size, mode):
    eval_dataloader = DataLoader(
        list(zip(val_data["x"], val_data["y"])),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
    )

    pbar = tqdm.tqdm(enumerate(eval_dataloader))
    accuracies = []
    for step, data in pbar:
        x, y = data
        x_ = tok(
            list(x),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor(y, device=DEVICE)
        with torch.inference_mode():

            if mode == "pimped_bert":
                eval_logits = model(x_) 
            else:
                eval_logits = model(**x_).logits

            total_acc = get_acc(eval_logits, y_)
            accuracies.append(total_acc)

    return np.mean(accuracies)


def ft_bert(model, tok, x, y, val, mode, batch_size=32, saving_path=""):
    model = copy.deepcopy(model).to(DEVICE)
    
    if mode != "pimped_bert":
        optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode), lr=1e-4)
    else:
        model = SurgicalFineTuningBert(model).to(DEVICE)
        all_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(all_params, lr=1e-4)

    train_dataloader = DataLoader(
        list(zip(x, y)), batch_size=batch_size, shuffle=True, collate_fn=None
    )
    eval_dataloader = DataLoader(
        list(zip(x, y)), batch_size=batch_size, shuffle=True, collate_fn=None
    )

    print(train_dataloader)
    
    pbar = tqdm.tqdm(enumerate(train_dataloader))
    for step, data in pbar:

        x, y = data
        x_ = tok(
            list(x),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor(y, device=DEVICE)
        
        if mode == "pimped_bert":
            logits = model(x_) 
        else:
            logits = model(**x_).logits

        loss = get_loss(logits, y_)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if args.debug:
            break
        

        if step % 10 == 0:
            xval, yval = next(iter(eval_dataloader))

            x_val = tok(
            list(xval),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
            ).to(DEVICE)
            y_val = torch.tensor(yval, device=DEVICE)

            with torch.inference_mode():
                if mode == "pimped_bert":
                    eval_logits = model(x_val) 
                else:
                    eval_logits = model(**x_val).logits

                total_acc = get_acc(eval_logits, y_val)
        pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")


        if step % 200 == 0 and saving_path != "":
            val_acc = eval(model, tok, val, batch_size, mode)
            print(f"\n Validation accuracy: {val_acc}")
            # torch.save(
            #         {"model_state_dict": model.state_dict()},
            #         saving_path + "_val_acc_" + str(round(val_acc,2)) + f"_step_{step}.pt",
            #     )

    return model


def run_ft(
    models: List[str],
    train_datasets: List[str],
    val_datasets: List[str],
    train_percentages: List[int],
    val_percentages: List[int],
    modes: List[str],
    batch_size,
    n_train: int = 1000,
    n_val: int = 100
):
    results = {}

    train, val = get_train_val_datasets(
        train_datasets,
        val_datasets,
        train_percentages,
        val_percentages,
        n_train,
        n_val,
    )

    for model_name, mode in itertools.product(models, modes):
        print(f"Fine-tuning {model_name} on and mode={mode}")

        if mode not in ["first", "middle", "last", "all", "pimped_bert"]:
            raise ValueError(mode, "is not a valid argument for argument mode")

        model = None
        tokenizer = None
        if "amazon" in train_datasets[0]:
            model, tokenizer = get_model_and_tokenizer(
                model_name,
                transformers.AutoModelForSequenceClassification,
                num_labels=5,
            )
            # _, tokenizer = get_model_and_tokenizer(
            #     model_name, transformers.BertTokenizer, num_labels=5
            # )
        else:
            model, tokenizer = get_model_and_tokenizer(
                model_name,
                transformers.AutoModelForSequenceClassification,
                num_labels=2,
            )

        eval_only_str = (args.eval_only == 0) * "finetune_and_eval" + (
            args.eval_only == 1
        ) * "eval_only"

        description_str = "_".join(
            [
                model_name,
                "train",
                "-".join(train_datasets),
                "val",
                "-".join(val_datasets),
                "train_pct",
                "-".join([str(p) for p in train_percentages]),
                "val_pct",
                "-".join([str(p) for p in val_percentages]),
                mode,
                eval_only_str,
            ]
        )

        print(f"Begin training for {description_str}")

        for repeat in range(args.repeats):

            path_ckpt = f"results/ft/fine_tuned_{description_str}.pt"

            print(f"Beginning repeat #{repeat}")
            if args.path_ckpt is not None:
                ckpt = torch.load(args.path_ckpt)
                model.load_state_dict(ckpt["model_state_dict"])

            if repeat > 0:
                model = fine_tuned

            if args.eval_only == 0:
                fine_tuned = ft_bert(
                    model,
                    tokenizer,
                    train["x"],
                    train["y"],
                    val,
                    mode,
                    batch_size=batch_size,
                    saving_path=path_ckpt[:-3]
                )
                val_acc = eval(fine_tuned, tokenizer, val, batch_size, mode)
            else:
                val_acc = eval(model, tokenizer, val, batch_size, mode)

            results[description_str] = val_acc

            question = "ft"
            if not os.path.exists(f"results/{question}"):
                os.makedirs(f"results/{question}")

            if args.eval_only == 0:
                torch.save(
                    {"model_state_dict": fine_tuned.state_dict()},
                    path_ckpt,
                )

            print(results)

            for k_, v in results.items():
                with open(f"results/{question}/{k_}.json", "w") as f:
                    json.dump({"metric": v}, f)
            results = {}


if __name__ == "__main__":
    train_percentages = [int(k) for k in args.train_percentages.split(",")]
    val_percentages = [int(k) for k in args.val_percentages.split(",")]
    # run_ft(
    #     ["bert-med"],
    #     ["amazon_electronics", "amazon_video"],
    #     ["amazon_electronics", "amazon_video"],
    #     [95, 5],
    #     [95, 5],
    #     ["last"],
    #     100000,
    #     args.batch_size,
    #     args.n_train,
    #     args.n_val,
    # )
    run_ft(
        args.model.split(","),
        args.train_dataset.split(","),
        args.val_dataset.split(","),
        train_percentages,
        val_percentages,
        args.mode.split(","),
        args.batch_size,
        args.n_train,
        args.n_val
    )
