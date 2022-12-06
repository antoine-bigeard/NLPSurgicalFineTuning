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
parser.add_argument("--model", default="bert-tiny")
# parser.add_argument("--dataset", default=["amazon_videos"])
parser.add_argument("--train_percentages", default="95,5")
parser.add_argument("--val_percentages", default="95,5")
parser.add_argument("--train_dataset", default="amazon_electronics,amazon_video")
parser.add_argument("--val_dataset", default="amazon_electronics,amazon_video")
parser.add_argument("--mode", default="pimped_bert")
parser.add_argument("--base_model_ckpt", default=None)
parser.add_argument("--save_path_ckpt", default=None)
parser.add_argument("--load_path_ckpt", default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--eval_only", default=0, type=int)
parser.add_argument("--n_train", default=10000, type=int)
parser.add_argument("--n_val", default=100, type=int)
parser.add_argument("--n_epochs", default=5, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
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


def eval(model, tok, eval_dataloader, mode):

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


def ft_bert(
    model,
    optimizer,
    tok,
    train_dataloader,
    eval_dataloader,
    mode,
    saving_path="",
    n_epochs=5,
    description_str="",
):
    model = copy.deepcopy(model).to(DEVICE)

    print(f"Train samples: {len(train_dataloader)}")
    print(f"Val samples: {len(eval_dataloader)}")

    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(enumerate(train_dataloader), disable=True)
        print(f"Epoch {epoch}")
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

            if mode == "pimped_bert":
                model.normalize_alphas()

            if step % 100 == 0:
                val_acc = eval(model, tok, eval_dataloader, mode)
                # pbar.set_description(f"Fine-tuning val accuracy: {val_acc:.04f}")
                print(f"Fine-tuning val accuracy: {val_acc:.04f}")

                if mode == "pimped_bert":
                    alphas = model.get_alphas()
                    alphas_opti = torch.ones(len(alphas))
                    alphas_frozen = torch.zeros(len(alphas))
                    val_acc_opti = eval(
                        lambda x: model.forward_alphas(x, alphas=alphas_opti),
                        tok,
                        eval_dataloader,
                        mode,
                    )
                    val_acc_frozen = eval(
                        lambda x: model.forward_alphas(x, alphas=alphas_frozen),
                        tok,
                        eval_dataloader,
                        mode,
                    )

                    print(f"Accuracy opti only: {val_acc_opti:.04f}")
                    print(f"Accuracy frozen only: {val_acc_frozen:.04f}")
                    print("Alphas: ", alphas)

                    f = open(
                        f"src/results/ft/{description_str}.txt",
                        "a",
                    )

                    f.write(
                        "Epoch "
                        + str(epoch)
                        + ", Step "
                        + str(step)
                        + " | alphas: "
                        + str(alphas)
                        + " , accuracy: "
                        + str(round(val_acc, 4))
                        + "\n"
                    )
                    f.close()

        if saving_path != "" and saving_path is not None:

            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(saving_path, description_str + ".pt"),
            )
            with open(os.path.join("src/results/ft/", description_str + ".txt")) as f:
                f.write(
                    "Epoch "
                    + str(epoch)
                    + ", Step "
                    + str(step)
                    + " | Accuracy: "
                    + str(round(val_acc, 4))
                    + "\n"
                )

    return model


def run_ft(
    models: List[str],
    train_datasets: List[str],
    val_datasets: List[str],
    train_percentages: List[int],
    val_percentages: List[int],
    modes: List[str],
    batch_size,
    n_epochs,
    n_train: int = 1000,
    n_val: int = 100,
    base_model_ckpt=None,
    load_path_ckpt=None,
    save_path_ckpt=None,
    eval_only=0,
    learning_rate=1e-3,
):
    results = {}

    train_data, val_data = get_train_val_datasets(
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

        tokenizer = None
        if "amazon" in train_datasets[0]:
            model, tokenizer = get_model_and_tokenizer(
                model_name,
                transformers.AutoModelForSequenceClassification,
                num_labels=5,
            )
        else:
            model, tokenizer = get_model_and_tokenizer(
                model_name,
                transformers.AutoModelForSequenceClassification,
                num_labels=2,
            )

        eval_only_str = (eval_only == 0) * "finetune_and_eval" + (
            eval_only == 1
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

        # path_ckpt = f"results/ft/fine_tuned_{description_str}.pt"

        if mode == "pimped_bert":
            if base_model_ckpt is not None:
                ckpt = torch.load(base_model_ckpt)
                model.load_state_dict(ckpt["model_state_dict"])
            model = SurgicalFineTuningBert(bert_model=model).to(DEVICE)
            if load_path_ckpt is not None:
                ckpt = torch.load(load_path_ckpt)
                model.load_state_dict(ckpt["model_state_dict"])
            all_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        else:
            if load_path_ckpt is not None:
                ckpt = torch.load(load_path_ckpt)
                model.load_state_dict(ckpt["model_state_dict"])
            optimizer = torch.optim.Adam(
                parameters_to_fine_tune(model, mode), lr=learning_rate
            )

        train_dataloader = DataLoader(
            list(zip(train_data["x"], train_data["y"])), batch_size=batch_size
        )
        eval_dataloader = DataLoader(
            list(zip(val_data["x"], val_data["y"])), batch_size=batch_size
        )

        if eval_only == 0:
            fine_tuned = ft_bert(
                model,
                optimizer,
                tokenizer,
                train_dataloader,
                eval_dataloader,
                mode,
                save_path_ckpt,
                n_epochs,
                description_str=description_str,
            )
            val_acc = eval(fine_tuned, tokenizer, eval_dataloader, mode)
        else:
            val_acc = eval(model, tokenizer, eval_dataloader, mode)

        results[description_str] = val_acc

        question = "ft"
        if not os.path.exists(f"results/{question}"):
            os.makedirs(f"results/{question}")

        if eval_only == 0:
            torch.save(
                {"model_state_dict": fine_tuned.state_dict()},
                os.path.join(save_path_ckpt, description_str + ".pt"),
            )

        print(results)

        for k_, v in results.items():
            with open(f"results/{question}/{k_}.json", "w") as f:
                json.dump({"metric": v}, f)
        results = {}


if __name__ == "__main__":
    train_percentages = [int(k) for k in args.train_percentages.split(",")]
    val_percentages = [int(k) for k in args.val_percentages.split(",")]

    run_ft(
        models=["bert-med"],
        train_datasets=["amazon_electronics"],
        val_datasets=["amazon_electronics"],
        train_percentages=[100],
        val_percentages=[100],
        modes=["pimped_bert"],
        batch_size=8,
        n_epochs=10,
        n_train=10000,
        n_val=100,
        # base_model_ckpt="ckpts/bert-med_train_amazon_electronics_val_amazon_electronics_train_pct_100_val_pct_100_all_finetune_and_eval.pt",
        # load_path_ckpt="ckpts/bert-med_train_amazon_electronics_val_amazon_electronics_train_pct_100_val_pct_100_pimped_bert_finetune_and_eval.pt",
        save_path_ckpt="ckpts",
        eval_only=0,
        learning_rate=1e-3,
    )
    # python src/main.py --model bert-med --mode pimped_bert --train_dataset amazon_books --val_dataset amazon_books --train_percentages 100 --val_percentages 100 --batch_size 16 --n_train 10000 --n_val 100 --eval_only 0    run_ft(
    run_ft(
        models=args.model.split(","),
        train_datasets=args.train_dataset.split(","),
        val_datasets=args.val_dataset.split(","),
        train_percentages=train_percentages,
        val_percentages=val_percentages,
        modes=args.mode.split(","),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        base_model_ckpt=args.base_model_ckpt,
        load_path_ckpt=args.load_path_ckpt,
        save_path_ckpt=args.save_path_ckpt,
        eval_only=args.eval_only,
        learning_rate=args.lr,
    )
