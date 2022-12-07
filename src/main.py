import sys

sys.path.insert(0, ".")
from src.pimped_bert import *
from src.utils import count_parameters
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
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(10)
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="bert-tiny")
# parser.add_argument("--dataset", default=["amazon_videos"])
parser.add_argument("--train_percentages", default="95,5")
parser.add_argument("--val_percentages", default="95,5")
parser.add_argument("--train_dataset", default="amazon_electronics,amazon_video")
parser.add_argument("--val_dataset", default="amazon_electronics,amazon_video")
parser.add_argument("--mode", default="pimped_bert")
parser.add_argument("--base_model_ckpt", default=None)
parser.add_argument("--load_path_ckpt", default=None)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--device", default="cuda:1")
parser.add_argument("--eval_only", default=0, type=int)
parser.add_argument("--n_train", default=10000, type=int)
parser.add_argument("--n_val", default=100, type=int)
parser.add_argument("--n_epochs", default=5, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--idxs_alphas", default="0,0,0,1,1", type=str)
parser.add_argument("--val_freq", default=50, type=int)
args = parser.parse_args()


DEVICE = torch.device(args.device)

torch.manual_seed(123)


def parameters_to_fine_tune(model: nn.Module, mode: str, idxs_alphas=None):
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
        if "bert-small" in model.name_or_path:
            return (
                list(model.bert.encoder.layer[-1].parameters())
                + list(model.bert.pooler.parameters())
                + list(model.classifier.parameters())
            )
        return list(model.bert.encoder.layer[-2:].parameters())
    elif mode == "first":
        if "bert-small" in model.name_or_path:
            return list(model.bert.embeddings.parameters()) + list(
                model.bert.encoder.layer[0].parameters()
            )
        return list(model.bert.encoder.layer[:2].parameters())
    elif mode == "middle":
        if "bert-small" in model.name_or_path:
            return list(model.bert.encoder.layer[1:3].parameters())
        n_trans = len(model.bert.encoder.layer)
        return list(
            model.bert.encoder.layer[n_trans // 2 - 1 : n_trans // 2 + 1].parameters()
        )

    elif mode == "all_but_embeds_pooler" and idxs_alphas is not None:
        parameters = []
        for key, value in model.named_parameters():
            if "embeddings" not in key and "pooler" not in key:
                parameters.append(value)
        return parameters
    elif mode == "perso" and idxs_alphas is not None:
        parameters = []
        for i in idxs_alphas[:-1]:
            if i:
                parameters += list(model.bert.encoder.layer[i].parameters())
        if idxs_alphas[-1]:
            parameters += list(model.classifier.parameters())
        return parameters
    else:
        raise NotImplementedError()


def get_loss(logits: torch.tensor, targets: torch.tensor) -> torch.tensor:
    loss = nn.functional.cross_entropy(logits, targets)
    return loss


def get_acc(logits, targets):
    y = torch.argmax(logits, dim=-1) == targets
    y = y.type(torch.float)
    return torch.mean(y).item()


class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "alphas"):
            w = module.alphas.data
            # w = w.clamp(-1, 1)
            w = (w.sigmoid() / w.sigmoid().sum()).logit()
            module.alphas.data = w


clipper = WeightClipper()


def eval_model(model, tok, eval_dataloader, mode):

    pbar = tqdm.tqdm(enumerate(eval_dataloader), disable=True)
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
    n_epochs=5,
    description_str="",
    val_freq=50,
):
    model = model.to(DEVICE)

    log_dir = os.path.join("logs", description_str)
    if os.path.isdir(log_dir):
        existing_experiments = os.listdir(log_dir)
        if len(log_dir) > 0:
            last_version = max([exp.split("_")[-1] for exp in existing_experiments])
            log_dir = os.path.join(
                log_dir,
                f"version_{int(last_version)+1}",
            )
    else:
        log_dir = os.path.join(log_dir, "version_0")

    writer = SummaryWriter(log_dir=log_dir)

    print(f"Train samples: {len(train_dataloader)}")
    print(f"Val samples: {len(eval_dataloader)}")

    old_val_acc = 0
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        pbar = tqdm.tqdm(
            enumerate(train_dataloader),
            position=0,
            leave=True,
            disable=False,
            total=len(train_dataloader),
        )
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
            writer.add_scalar("loss/loss", loss, step + epoch * len(train_dataloader))
            writer.add_scalar(
                "acc/train", get_acc(logits, y_), step + epoch * len(train_dataloader)
            )
            loss.backward()
            optimizer.step()
            if mode == "pimped_bert":
                model.apply(clipper)
            optimizer.zero_grad()
            if args.debug:
                break

            #     model.normalize_alphas()

            if step % val_freq == 0:
                val_acc = eval_model(model, tok, eval_dataloader, mode)
                if old_val_acc < val_acc:
                    torch.save(
                        {"model_state_dict": model.state_dict()},
                        os.path.join(log_dir, "best_ckpt.pt"),
                    )
                    old_val_acc = val_acc
                    writer.add_scalar(
                        "val_acc_ckpt", val_acc, step + epoch * len(train_dataloader)
                    )
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    os.path.join(log_dir, f"last.pt"),
                )
                # pbar.set_description(f"Fine-tuning val accuracy: {val_acc:.04f}")
                # print(f"Fine-tuning val accuracy: {val_acc:.04f}")
                writer.add_scalar(
                    "acc/val", val_acc, step + epoch * len(train_dataloader)
                )

                if mode == "pimped_bert":
                    alphas = model.get_alphas()
                    alphas_opti = torch.ones(len(alphas))
                    alphas_frozen = torch.zeros(len(alphas))
                    val_acc_opti = eval_model(
                        lambda x: model.forward_alphas(x, alphas=alphas_opti),
                        tok,
                        eval_dataloader,
                        mode,
                    )
                    val_acc_frozen = eval_model(
                        lambda x: model.forward_alphas(x, alphas=alphas_frozen),
                        tok,
                        eval_dataloader,
                        mode,
                    )

                    writer.add_scalar(
                        "acc/val_opti",
                        val_acc_opti,
                        step + epoch * len(train_dataloader),
                    )
                    writer.add_scalar(
                        "acc/val_frozen",
                        val_acc_frozen,
                        step + epoch * len(train_dataloader),
                    )

                    # print(f"Accuracy opti only: {val_acc_opti:.04f}")
                    # print(f"Accuracy frozen only: {val_acc_frozen:.04f}")
                    # print("Alphas: ", alphas)
                    for i, a in enumerate(alphas):
                        writer.add_scalar(
                            f"alphas/{i}", a, step + epoch * len(train_dataloader)
                        )

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

                else:
                    f = open(
                        f"src/results/ft/{description_str}.txt",
                        "a",
                    )

                    f.write(
                        "Epoch "
                        + str(epoch)
                        + ", Step "
                        + str(step)
                        + " , accuracy: "
                        + str(round(val_acc, 4))
                        + "\n"
                    )
                    f.close()

        # torch.save(
        #     {"model_state_dict": model.state_dict()},
        #     os.path.join(log_dir, f"last.pt"),
        # )
        f = open(
            f"src/results/ft/{description_str}.txt",
            "a",
        )
        f.write(
            "End of epoch "
            + str(epoch)
            + " | Accuracy: "
            + str(round(val_acc, 4))
            + "\n"
        )
        f.close()

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
    eval_only=0,
    learning_rate=1e-3,
    idxs_alphas=None,
    val_freq=50,
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

        if mode not in [
            "first",
            "middle",
            "last",
            "all",
            "pimped_bert",
            "perso",
            "all_but_embeds_pooler",
        ]:
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
                parameters_to_fine_tune(model, mode, idxs_alphas=idxs_alphas),
                lr=learning_rate,
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
                n_epochs,
                description_str=description_str,
                val_freq=val_freq,
            )
            val_acc = eval_model(fine_tuned, tokenizer, eval_dataloader, mode)
        else:
            model = model.to(DEVICE)
            val_acc = eval_model(model, tokenizer, eval_dataloader, mode)

        results[description_str] = val_acc

        question = "ft"
        if not os.path.exists(f"results/{question}"):
            os.makedirs(f"results/{question}")

        print(results)

        for k_, v in results.items():
            with open(f"results/{question}/{k_}.json", "w") as f:
                json.dump({"metric": v}, f)
        results = {}


if __name__ == "__main__":
    train_percentages = [int(k) for k in args.train_percentages.split(",")]
    val_percentages = [int(k) for k in args.val_percentages.split(",")]
    idxs_alphas = [int(k) for k in args.idxs_alphas.split(",")]

    # run_ft(
    #     models=["bert-small"],
    #     train_datasets=["amazon_books", "amazon_video"],
    #     val_datasets=["amazon_books", "amazon_video"],
    #     train_percentages=[80, 20],
    #     val_percentages=[80, 20],
    #     modes=["all"],
    #     batch_size=128,
    #     n_epochs=4,
    #     n_train=200,
    #     n_val=10,
    #     # base_model_ckpt="logs/all_fine_tuned_books/ckpt_epoch_0_step_1250.pt",
    #     load_path_ckpt="logs/all_fine_tuned_books/ckpt_epoch_0_step_1250.pt",
    #     eval_only=1,
    #     learning_rate=1e-4,
    #     idxs_alphas=[1, 1, 1, 1, 1],
    #     val_freq=50,
    # )
    # python src/main.py --model bert-med --mode pimped_bert --train_dataset amazon_books --val_dataset amazon_books --train_percentages 100 --val_percentages 100 --batch_size 16 --n_train 10000 --n_val 100 --eval_only 0    run_ft(
    # run_ft(
    #     models=["bert-med"],
    #     train_datasets=["amazon_electronics"],
    #     val_datasets=["amazon_electronics"],
    #     train_percentages=[100],
    #     val_percentages=[100],
    #     modes=["pimped_bert"],
    #     batch_size=128,
    #     n_epochs=10,
    #     n_train=10000,
    #     n_val=10,
    #     # base_model_ckpt="ckpts/bert-med_train_amazon_electronics_val_amazon_electronics_train_pct_100_val_pct_100_all_finetune_and_eval.pt",
    #     # load_path_ckpt="ckpts/bert-med_train_amazon_electronics_val_amazon_electronics_train_pct_100_val_pct_100_pimped_bert_finetune_and_eval.pt",
    #     eval_only=0,
    #     learning_rate=1e-3,
    #     idxs_alphas=None,
    #     val_freq=50,
    # )
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
        eval_only=args.eval_only,
        learning_rate=args.lr,
        idxs_alphas=args.idxs_alphas,
        val_freq=int(args.val_freq),
    )
