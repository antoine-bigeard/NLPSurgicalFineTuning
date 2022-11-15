import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from src.data.datamodule.datamodule import MyDataModule
from src.model.model.pretrained_model import PretrainedTorchModel
from src.model.lit_model.lit_model import MyLitModel
from src.utils import read_yaml_config_file

from src.utils import *

import argparse
import copy
import os
import torch
import torch.nn as nn
import transformers
import tqdm
import yaml

DEVICE = torch.device('cuda')

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
    if mode == 'all':
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
    all_x = tok(x, return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok([x[i] for i in batch], return_tensors='pt', padding=True, truncation=True, max_length=100).to(DEVICE)
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
            pbar.set_description(f'Fine-tuning acc: {total_acc:.04f}')
            if total_acc > 0.75:
                break
    return model

def run_ft(models: List[str], datasets: List[str], percentages: List[int], val_dataset: str, modes: List[str], n_train: int = 20000, n_val: int = 1000):
    results = {}

    train, val = get_dataset(datasets, percentages, val_dataset, n_train, n_val)

    for model_name, mode in itertools.product(models, modes):
        print(f'Fine-tuning {model_name} on and mode={mode}')
        model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForSequenceClassification, num_labels=5)
            
        for repeat in range(args.repeats):
            if repeat > 0:
                print(f'Beginning repeat #{repeat}')
                fine_tuned = ft_bert(model, tokenizer, train['x'][:k*5], train['y'][:k*5], mode)
                val_acc = eval(fine_tuned, tokenizer, val)
                results['_'.join([model_name, dataset, str(k), mode])] = val_acc

            print(results)
            question = 'ft'
            if not os.path.exists(f'results/{question}'):
                os.makedirs(f'results/{question}')

            for k_, v in results.items():
                with open(f'results/{question}/{k_}.json', 'w') as f:
                    json.dump({'metric': v}, f)
            results = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_config",
        help="config path that contains config for data, models, training.",
        default="/home/ec2-user/repo/team_b/config/config.yaml",
        required=False,
    )
    parser.add_argument(
        "--mode",
        help="mode fit, test or predict",
        default="fit",
        required=False,
    )

    args = parser.parse_args()

    path_config = args.path_config
    mode = args.mode

    config = read_yaml_config_file(path_config)
    checkpoint_path = config.get("checkpoint_path")
    conf_datamodule = config.get("datamodule")
    conf_model = config.get("model")
    conf_pretrained_model = config.get("pretrained_model")
    conf_lit_model = config.get("lit_model")
    conf_trainer = config.get("trainer")
    name_exp = config.get("name_experiment")
    conf_ts_board = config.get("tensorboard_logs")
    conf_checkpoint_callback = config.get("checkpoint_callback")

    tsboard_logger = TensorBoardLogger(
        conf_ts_board["save_dir"],
        conf_ts_board["name"],
    )

    logs_folder = os.path.join(
        tsboard_logger.save_dir,
        tsboard_logger.name,
        f"version_{tsboard_logger.version}",
    )
    os.makedirs(logs_folder, exist_ok=True)

    datamodule = MyDataModule(
        transforms_img=transforms_data,
        transform_window={},
        transforms_meta=None,
        categories=categories,
        path_logs=logs_folder,
        **conf_datamodule,
    )

    num_classes = len(datamodule.selected_categories)

    if conf_lit_model["use_torch_model"]:
        model = PretrainedTorchModel(**conf_pretrained_model, num_classes=num_classes)

    lit_model = MyLitModel(
        model=model,
        learning_rate=conf_lit_model["learning_rate"],
    )

    if checkpoint_path is not None:
        lit_model.load_from_checkpoint(checkpoint_path)

    with open(os.path.join(logs_folder, "config.yaml"), "w") as dst:
        yaml.dump(config, dst)

    early_stop_callback = EarlyStopping(
        monitor="train/loss",
        min_delta=0.01,
        patience=10,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logs_folder, "checkpoints"),
        verbose=True,
        **conf_checkpoint_callback,
    )

    trainer = pl.Trainer(
        logger=tsboard_logger,
        callbacks=[early_stop_callback],
        **conf_trainer,
        enable_progress_bar=True,
    )

    if mode == "fit":
        trainer.fit(lit_model, datamodule)

    elif mode == "test":
        trainer.test(lit_model, datamodule)

    elif mode == "predict":
        trainer.predict(lit_model, datamodule)

    else:
        raise ValueError("Please give a valid mode: fit, test or predict")
