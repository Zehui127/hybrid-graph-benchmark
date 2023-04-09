import pytorch_lightning as pl
import torch
import numpy as np
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR


class ModelWrapper(pl.LightningModule):

    def __init__(
            self,
            model,
            learning_rate=5e-4,
            epochs=200,
            optimizer=None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch, batch.y
        mask = batch.train_mask
        y_hat = self.forward(x)
        loss = self.loss(y_hat[mask], y[mask])
        acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=3)
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
        acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=3)
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
        acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=3)
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
