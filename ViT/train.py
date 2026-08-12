import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset
from tqdm import tqdm
import wandb
import seaborn as sns

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda"
            if config.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        if self.config.wandb_monitor:
            wandb.init(project = self.config.project_name,
                        name = self.config.run_name,
                        config = vars(config))

        if self.config.monitor == "val_acc":
            self.best_metric = -float("inf")
        else:
            self.best_metric = float("inf")

    def train(self, model, dataset):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
        )
        start_epoch = 0
        if self.config.load_from_checkpoint:
            start_epoch = self._load_checkpoint(
                model,
                optimizer,
            )

        for epoch in range(start_epoch, self.config.epochs):

            print(f"\nEpoch [{epoch+1}/{self.config.epochs}]")

            train_loss, train_acc = self._train_epoch(
                model,
                dataset.train_loader,
                criterion,
                optimizer,
            )

            val_loss, val_acc, _, _ = self._validate_epoch(
                model,
                dataset.test_loader,
                criterion,
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if self.config.wandb_monitor:
                wandb.log({"epoch" : epoch + 1,
                        "train/loss" : train_loss,
                        "train/accuracy" : train_acc,
                        "test/loss" : val_loss,
                        "test/accuracy": val_acc})
            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            if self.config.save_last_model:
                self._save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    "last.pt",
                )

            if self.config.save_best_model:

                metric = (
                    val_acc
                    if self.config.monitor == "val_acc"
                    else val_loss
                )

                improved = (
                    metric > self.best_metric
                    if self.config.monitor == "val_acc"
                    else metric < self.best_metric
                )

                if improved:
                    self.best_metric = metric

                    self._save_checkpoint(
                        model,
                        optimizer,
                        epoch + 1,
                        "best.pt",
                    )

                    if self.config.wandb_monitor:
                        wandb.log({
                            "best_val_accuracy": val_acc,
                            "best_val_loss" : val_loss
                        })

        _ = self._load_checkpoint(model, optimizer, best = True)

        if self.config.save_plots:
            self._plot_history()
            self._plot_confusion_matrix(model, dataset, criterion)
        
        if self.config.save_csv:
            self._save_csv()

        if self.config.wandb_monitor:
            self._upload_models_to_wandb()
            wandb.finish()
        
        


        return self.history

    def _train_epoch(
        self,
        model,
        loader,
        criterion,
        optimizer,
    ):

        model.train()

        running_loss = 0
        correct = 0
        total = 0
        loop = tqdm(loader)
        for images, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(
                loss=running_loss / (loop.n + 1),
                acc=100 * correct / total,
            )
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _validate_epoch(
        self,
        model,
        loader,
        criterion,
    ):
        model.eval()
        running_loss = 0
        correct = 0
        total = 0
        actual, predict =[], []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            actual.append(predicted)
            predict.append(labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc, actual, predict
    def _save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        filename,
    ):
        os.makedirs(
            self.config.checkpoint_path,
            exist_ok=True,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(
                self.config.checkpoint_path,
                filename,
            ),
        )

    def _load_checkpoint(
        self,
        model,
        optimizer,
        best = False
    ):

        checkpoint = torch.load(
            os.path.join(
                self.config.checkpoint_path,
                "best.pt" if best else "last.pt",
            ),
            map_location=self.device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        print("Checkpoint Loaded")
        return checkpoint["epoch"]

    def _upload_models_to_wandb(self):

        best_path = os.path.join(
            self.config.checkpoint_path,
            "best.pt"
        )

        last_path = os.path.join(
            self.config.checkpoint_path,
            "last.pt"
        )


        # Upload best model
        if os.path.exists(best_path):

            best_artifact = wandb.Artifact(
                name="best-model",
                type="model",
                description="Best validation performance model"
            )

            best_artifact.add_file(best_path)

            wandb.log_artifact(best_artifact)


        # Upload last model
        if os.path.exists(last_path):

            last_artifact = wandb.Artifact(
                name="last-model",
                type="model",
                description="Final epoch model"
            )

            last_artifact.add_file(last_path)

            wandb.log_artifact(last_artifact)
    
    def _plot_history(self):
        os.makedirs(
            self.config.plot_path,
            exist_ok=True,
        )
        epochs = range(
            1,
            len(self.history["train_loss"]) + 1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(
            epochs,
            self.history["train_loss"],
            label="Train",
        )
        plt.plot(
            epochs,
            self.history["val_loss"],
            label="Validation",
        )
        loss_path = os.path.join(self.config.plot_path, "loss.png")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(loss_path)

        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(
            epochs,
            self.history["train_acc"],
            label="Train",
        )
        plt.plot(
            epochs,
            self.history["val_acc"],
            label="Validation",
        )
        acc_path = os.path.join(self.config.plot_path,"accuracy.png")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy")
        plt.legend()
        plt.savefig(acc_path)
        plt.close()

        if self.config.wandb_monitor:
            wandb.log({
                "loss_curve": wandb.Image(loss_path),
                "acc_curve" : wandb.Image(acc_path)
            })

    def _plot_confusion_matrix(self, model, dataset, criterion):
        idToClass = dataset.idToclasses
        _, _, actual, predict = self._validate_epoch(model, dataset.test_loader, criterion)

        actual = [i for batch in actual for i in batch]
        predict = [i for batch in predict for i in batch]
        cm = [[0 for _ in range(len(idToClass))] for _ in range(len(idToClass))]
        for i, j in zip(actual, predict):
            cm[i-1][j-1] += 1
        cm = torch.tensor(cm)
        class_names = [k for k in idToClass.keys()]

        plt.figure(figsize = (7, 6))

        sns.heatmap(
            cm,
            annot = True,
            fmt = "d",
            cmap = "Blues",
            xticklabels = class_names,
            yticklabels = class_names
        )


        confusion_mat_path = os.path.join(self.config.plot_path, "confusion_mat.png")
        plt.xlabel("predicted_labels")
        plt.ylabel("Actual labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_mat_path)
        plt.close()

        if self.config.wandb_monitor:
            wandb.log(
                {
                    "confussion_mat" : wandb.Image(confusion_mat_path)
                }
            )
    
    def _save_csv(self):
        df = pd.DataFrame([self.history])
        df.to_csv(self.config.csv_path)
        if self.config.wandb_monitor:
            csv_artifact = wandb.Artifact("Csv_log", type = "dataset", description = "log csv file")
            csv_artifact.add_file(self.config.csv_path)
            wandb.log_artifact(csv_artifact)

        

if __name__ == "__main__":
    from model import AlexNet
    from dataset import Dataset
    from config import DatasetConfig, TrainConfig

    dataset_config = DatasetConfig()
    train_config = TrainConfig()
    model = AlexNet()
    loader = Dataset(dataset_config)

    trainer = Trainer(train_config)

    trainer.train(model, loader)
