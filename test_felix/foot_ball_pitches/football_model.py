import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import Callback
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from factory import Classifier


class FootballPitchModel(pl.LightningModule):
    """
    LightningModule for training and evaluating a football pitch classifier.

    Args:
        num_classes (int): Number of classes for classification (default: 2).
        ckpt_path (str): Path to the pretrained encoder checkpoint.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for the optimizer.
        b1 (float): Beta1 parameter for the Adam optimizer.
        b2 (float): Beta2 parameter for the Adam optimizer.
    """

    def __init__(
        self, num_classes=2, ckpt_path=None, lr=1e-4, wd=1e-4, b1=0.9, b2=0.999
    ):  # noqa: PLR0913
        super().__init__()
        self.save_hyperparameters()
        if num_classes > 2:
            self.model = Classifier(num_classes=num_classes, ckpt_path=ckpt_path)
        else:
            self.model = Classifier(num_classes=1, ckpt_path=ckpt_path)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8.0]))
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def forward(self, datacube):
        """
        Forward pass through the classifier.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits from the classifier.
        """

        waves = torch.tensor(
            [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
        )
        gsd = torch.tensor(10.0)

        return self.model(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            }
        )

        return self.model(x)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=self.hparams.lr * 0.1, last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        """
        Perform a shared step for both training and validation.

        Args:
            batch (tuple): A batch of data containing (inputs, labels).
            batch_idx (int): The index of the batch.
            phase (str): The phase ('train' or 'val').

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        labels = batch["label"]  # .long()
        logits = self(batch).squeeze()
        loss = self.loss_fn(logits, labels)
        score = self.accuracy(logits, labels)
        auroc = self.auroc(logits, labels)
        precision = self.precision(logits, labels)
        recall = self.recall(logits, labels)
        f1 = self.f1(logits, labels)

        self.log(
            f"{phase}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/accuracy",
            score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/precision",
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/recall",
            recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if phase == "val":
            return {"loss": loss, "preds": logits, "targets": labels}
        if phase == "train":
            return {"loss": loss, "preds": logits, "targets": labels}

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.

        Args:
            batch (tuple): A batch of training data containing (inputs, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.

        Args:
            batch (tuple): A batch of validation data containing (inputs, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        outputs = self.shared_step(batch, batch_idx, "val")
        return outputs


class ConfusionMatrixCallback(Callback):
    def __init__(self, task="binary", num_classes=2):
        """
        Initialize the confusion matrix callback.

        Args:
            task (str): The task type ("binary" or "multiclass").
            num_classes (int): The number of classes (for binary, it's 2).
        """
        super().__init__()
        self.confmat = ConfusionMatrix(task=task, num_classes=num_classes)
        self.preds_val = []
        self.targets_val = []
        self.preds_train = []
        self.targets_train = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        Collect predictions and targets at the end of each validation batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule.
            outputs: The outputs from the validation step.
            batch: The validation batch.
            batch_idx: The index of the batch.
        """
        self.preds_val.append(outputs["preds"].detach().cpu())
        self.targets_val.append(outputs["targets"].detach().cpu())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Collect predictions and targets at the end of each training batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule.
            outputs: The outputs from the training step.
            batch: The training batch.
            batch_idx: The index of the batch.
        """
        self.preds_train.append(outputs["preds"].detach().cpu())
        self.targets_train.append(outputs["targets"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute and log the confusion matrix at the end of the validation epoch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule.
        """
        # Concatenate all predictions and targets
        preds = torch.cat(self.preds_val, dim=0)
        targets = torch.cat(self.targets_val, dim=0)

        # Compute confusion matrix
        confusion_matrix = self.confmat(preds, targets)

        # Log or save the confusion matrix
        print(
            f"VAL: Confusion Matrix (Epoch {trainer.current_epoch}):\n{confusion_matrix}"
        )
        # pl_module.log("confusion_matrix", confusion_matrix)

        # Clear the stored predictions and targets for the next epoch
        self.preds_val = []
        self.targets_val = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Compute and log the confusion matrix at the end of the training epoch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule.
        """
        # Concatenate all predictions and targets
        preds = torch.cat(self.preds_train, dim=0)
        targets = torch.cat(self.targets_train, dim=0)

        # Compute confusion matrix
        confusion_matrix = self.confmat(preds, targets)

        # Log or save the confusion matrix
        print(
            f"TRAIN: Confusion Matrix (Epoch {trainer.current_epoch}):\n{confusion_matrix}"
        )
        # pl_module.log("confusion_matrix", confusion_matrix)

        # Clear the stored predictions and targets for the next epoch
        self.preds_train = []
        self.targets_train = []
