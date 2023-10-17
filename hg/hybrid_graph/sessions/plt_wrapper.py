import pytorch_lightning as pl
import torch
import numpy as np
from torchmetrics.functional import accuracy
from torchmetrics import MeanAbsoluteError, Accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from ogb.nodeproppred import Evaluator

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
        if dataset_info['info']["is_regression"]:
            self.loss = torch.nn.MSELoss()
            self.train_acc = MeanAbsoluteError()
            self.val_acc = MeanAbsoluteError()
            self.test_acc = MeanAbsoluteError()
        elif dataset_info['info']["is_edge_pred"]:
            self.loss = torch.nn.BCEWithLogitsLoss()
            self.train_acc = Accuracy(task='binary')
            self.val_acc = Accuracy(task='binary')
            self.test_acc = Accuracy(task='binary')
        elif dataset_info['type']=='OBG':
            name = dataset_info['name']
            self.evaluator = Evaluator(name=f'ogbn-{name}')
            self.train_acc = lambda y_true, y_pred: self.evaluator.eval({'y_true': y_true,
                                                                    'y_pred': y_pred})['acc']
            self.val_acc = lambda y_true, y_pred: self.evaluator.eval({'y_true': y_true,
                                                                    'y_pred': y_pred})['acc']
            self.test_acc = lambda y_true, y_pred: self.evaluator.eval({'y_true': y_true,
                                                                    'y_pred': y_pred})['acc']
            self.loss = torch.nn.functional.nll_loss
        else:
            self.train_acc = Accuracy(task='multiclass',top_k=1,num_classes=dataset_info['info']["num_classes"])
            self.val_acc = Accuracy(task='multiclass',top_k=1,num_classes=dataset_info['info']["num_classes"])
            self.test_acc = Accuracy(task='multiclass',top_k=1,num_classes=dataset_info['info']["num_classes"])
            self.loss = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.dataset_info = dataset_info['info']

    def forward(self, x, *args, **kwargs):
        return self.model(x, args, kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch, batch.y
        y = batch.train_label.float() if self.dataset_info["is_edge_pred"] else y
        mask = batch.train_mask
        args = [batch.train_edge_index, batch.train_edge_label_index] if self.dataset_info["is_edge_pred"] else []
        y_hat = self.forward(x,*args)
        loss = self.loss(y_hat,y) if self.dataset_info["is_edge_pred"] else self.loss(y_hat[mask], y.squeeze(1)[mask])
        #acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=self.dataset_info["num_classes"]) if not self.dataset_info["is_regression"] else self.val_acc(y_hat[mask], y[mask])
        acc = self.train_acc(y_hat, y) if self.dataset_info["is_edge_pred"] else self.train_acc(y[mask],y_hat.argmax(dim=-1, keepdim=True)[mask])
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
        y = batch.val_label.float() if self.dataset_info["is_edge_pred"] else y
        mask = batch.val_mask
        args = [batch.val_edge_index, batch.val_edge_label_index] if self.dataset_info["is_edge_pred"] else []
        y_hat = self.forward(x,*args)
        # print(f"y_hat shape: {y_hat[mask].shape}")
        # print(f"y shape: {y[mask].shape}")
        # print(f"y shape squeeze: {y[mask].squeeze(1).shape}")
        #print(f"y_hat shape: {y_hat.shape}; y shape: {y.shape}")
        loss = self.loss(y_hat,y) if self.dataset_info["is_edge_pred"] else self.loss(y_hat[mask], y.squeeze(1)[mask])
        #acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=self.dataset_info["num_classes"]) if not self.dataset_info["is_regression"] else self.val_acc(y_hat[mask], y[mask])
        acc = self.train_acc(y_hat, y) if self.dataset_info["is_edge_pred"] else self.train_acc(y[mask],y_hat.argmax(dim=-1, keepdim=True)[mask])
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
        y = batch.test_label.float() if self.dataset_info["is_edge_pred"] else y
        mask = batch.test_mask
        args = [batch.test_edge_index, batch.test_edge_label_index] if self.dataset_info["is_edge_pred"] else []
        y_hat = self.forward(x,*args)
        loss = self.loss(y_hat,y) if self.dataset_info["is_edge_pred"] else self.loss(y_hat[mask], y.squeeze(1)[mask])
        #acc = accuracy(y_hat[mask], y[mask],task='multiclass',top_k=1,num_classes=self.dataset_info["num_classes"]) if not self.dataset_info["is_regression"] else self.val_acc(y_hat[mask], y[mask])
        acc = self.train_acc(y_hat, y) if self.dataset_info["is_edge_pred"] else self.train_acc(y[mask],y_hat.argmax(dim=-1, keepdim=True)[mask])
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
