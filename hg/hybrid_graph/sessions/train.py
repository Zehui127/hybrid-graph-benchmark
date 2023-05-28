from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from .plt_wrapper import ModelWrapper


def train(
        model,
        train_loader, val_loader,
        optimizer,
        learning_rate,
        plt_trainer_args,
        save_path,
        dataset_info,
        seed
        ):
    pl.seed_everything(seed, workers=True)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_acc",
        mode="max",
        filename="best",
        dirpath=save_path,
        save_last=True,
    )
    plt_trainer_args['callbacks'] = [checkpoint_callback]
    if save_path is not None:
        plt_trainer_args['default_root_dir'] = f'./lightning_logs/{save_path.split("checkpoints/")[1]}'
    plt_model = ModelWrapper(
        model,
        dataset_info,
        learning_rate=learning_rate,
        epochs=plt_trainer_args['max_epochs'],
        optimizer=optimizer)
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.fit(plt_model, train_loader, val_loader)
