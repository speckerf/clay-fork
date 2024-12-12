import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from factory import Classifier


class SMDsModel(pl.LightningModule):
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
        self.model = Classifier(num_classes=num_classes, ckpt_path=ckpt_path)
        self.loss_fn = nn.CrossEntropyLoss()

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

        self.log(
            f"{phase}/loss",
            loss,
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
