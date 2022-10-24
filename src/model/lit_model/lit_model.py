import pytorch_lightning as pl
import torchvision.models as models
import torch
import torch.nn.functional as F
from torchmetrics import (
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
    MetricCollection,
)
import numpy as np
import torchmetrics.functional as tm_metrics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from src.utils import read_yaml_config_file

from src.utils import categories


class MyLitModel(pl.LightningModule):
    def __init__(self, model: nn.Module = None, learning_rate: float = 2e-4):
        """Initiate the lightning model class.

        Args:
            model (nn.Module, optional): model used for the classification. Defaults to None.
            learning_rate (float, optional): learning rate for the training. Defaults to 2e-4.
        """
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = self.model.num_classes

        self.train_metrics = MetricCollection(
            Accuracy(), Precision(), F1Score(), Recall(), prefix="train/"
        )
        self.val_metrics = MetricCollection(
            Accuracy(), Precision(), F1Score(), Recall(), prefix="val/"
        )
        self.test_metrics = MetricCollection(
            Accuracy(), Precision(), F1Score(), Recall(), prefix="test/"
        )
        self.train_confusion_mat = ConfusionMatrix(
            self.num_classes,
            normalize="true",
        )
        self.val_confusion_mat = ConfusionMatrix(
            self.num_classes,
            normalize="true",
        )
        self.test_confusion_mat = ConfusionMatrix(
            self.num_classes,
            normalize="true",
        )

    def forward(self, x):
        """Forward method of the model."""
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Describe what to do on a training step.

        Args:
            batch (torch.Tensor): batch to process.
            batch_idx (int): batch index.

        Returns:
            dict: dictionnary containing the loss, the predictions and the targets (groud-truth).
        """
        imgs, labels = batch
        logits = self(imgs)
        loss = F.nll_loss(logits, torch.argmax(labels, dim=1))

        # training metrics
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(labels, dim=1)

        metrics = self.train_metrics(preds, targets)

        self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True)
        self.log_dict(metrics, on_step=True, on_epoch=False, logger=True)

        return {"loss": loss, "preds": preds, "target": targets}

    def training_epoch_end(self, outputs: dict) -> None:
        """Describe what to do on the end of a training epoch. Here, log information to tensorboard.

        Args:
            outputs (dict): outputs of the training steps.
        """
        preds = torch.cat([output["preds"] for output in outputs], dim=0)
        targets = torch.cat([output["target"] for output in outputs], dim=0)

        conf_mat = tm_metrics.confusion_matrix(
            preds, targets, num_classes=self.num_classes
        )

        df_cm = pd.DataFrame(
            conf_mat.cpu().numpy(),
            index=range(self.num_classes),
            columns=range(self.num_classes),
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure(
            "train/confusion_matrix", fig_, self.current_epoch
        )

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Same method as training_step but for validation."""
        imgs, labels = batch
        logits = self(imgs)
        loss = F.nll_loss(logits, torch.argmax(labels, dim=1))

        # training metrics
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(labels, dim=1)

        metrics = self.val_metrics(preds, targets)

        self.log("val/loss", loss, on_step=True, on_epoch=False, logger=True)
        self.log_dict(metrics, on_step=True, on_epoch=False, logger=True)

        return {"loss": loss, "preds": preds, "target": targets}

    def testing_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Same method as testing_step but for testing."""
        imgs, labels = batch
        logits = self(imgs)
        # loss = F.cross_entropy(logits, labels)
        loss = F.nll_loss(logits, torch.argmax(labels, dim=1))

        # training metrics
        preds = torch.argmax(logits, dim=1)
        targets = torch.argmax(labels, dim=1)

        metrics = self.test_metrics(preds, targets)

        self.log("test/loss", loss, on_step=True, on_epoch=False, logger=True)
        self.log_dict(metrics, on_step=True, on_epoch=False, logger=True)

        return {"loss": loss, "preds": preds, "target": targets}

    def configure_optimizers(self):
        """Configure optimizers to use for the training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
