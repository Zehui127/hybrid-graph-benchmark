import pytorch_lightning as pl
import torch
import numpy as np
from torchmetrics.functional import accuracy
from torchmetrics import MeanAbsoluteError, Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


class ModelWrapper(pl.LightningModule):

    def __init__(
            self,
            model,
            dataset_info,
            learning_rate=5e-4,
            epochs=200,
            optimizer=None
            ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        if dataset_info["is_regression"]:
            self.loss = torch.nn.MSELoss()
            self.train_acc = MeanAbsoluteError()
            self.val_acc = MeanAbsoluteError()
            self.test_acc = MeanAbsoluteError()
        else:
            self.train_acc = Accuracy(task='multiclass',top_k=1,num_classes=dataset_info["num_classes"])
            self.val_acc = Accuracy(task='multiclass',top_k=1,num_classes=dataset_info["num_classes"])
            self.test_acc = Accuracy(task='multiclass',top_k=1,num_classes=dataset_info["num_classes"])
            self.loss = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.dataset_info = dataset_info

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch, batch.y
        mask = batch.train_mask
        y_hat = self.forward(x)
        loss = self.loss(y_hat[mask], y[mask])
        #acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=self.dataset_info["num_classes"]) if not self.dataset_info["is_regression"] else self.train_acc(y_hat[mask], y[mask])
        acc = self.train_acc(y_hat[mask], y[mask])
        # loss
        self.log_dict(
            {"loss": loss},
            on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # acc
        self.log_dict(
            {"acc": acc},
            on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch, batch.y
        mask = batch.val_mask
        y_hat = self.forward(x)
        loss = self.loss(y_hat[mask], y[mask])
        #acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=self.dataset_info["num_classes"]) if not self.dataset_info["is_regression"] else self.val_acc(y_hat[mask], y[mask])
        acc = self.val_acc(y_hat[mask], y[mask])
        # val_loss
        self.log_dict(
            {"val_loss": loss},
            on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # val acc
        self.log_dict(
            {"val_acc": acc},
            on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch, batch.y
        mask = batch.test_mask
        y_hat = self.forward(x)
        loss = self.loss(y_hat[mask], y[mask])
        #acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=self.dataset_info["num_classes"]) if not self.dataset_info["is_regression"] else self.test_acc(y_hat[mask], y[mask])
        acc = self.test_acc(y_hat[mask], y[mask])
        # val_loss
        self.log_dict(
            {"test_loss": loss},
            on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # val acc
        self.log_dict(
            {"test_acc": acc},
            on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ['sgd_warmup', 'sgd']:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=0.0005,
                nesterov=True)
            if self.optimizer == 'sgd':
                scheduler = CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=0.0)
            elif self.optimizer == 'sgd_warmup':
                # mostly only useful for large scale image datasets
                from warmup_scheduler import GradualWarmupScheduler
                scheduler = CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=0.0)
                scheduler = GradualWarmupScheduler(
                    opt,
                    multiplier=2,
                    total_epoch=5,
                    after_scheduler=scheduler)
        return {
            "optimizer": opt,
            "lr_scheduler":  scheduler}
