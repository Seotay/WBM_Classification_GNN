import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import torch
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, epochs, optimizer, early_stopping, device):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping = early_stopping

        # loss history
        self.train_losses, self.val_losses = [], []
        self.train_f1s, self.val_f1s = [], []

    def training(self):

        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_one_epoch()
            val_loss, val_metrics = self.evaluate(data_loader=self.val_loader, loader_name="Validation Evaluating...",)

            # loss
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1s.append(train_metrics["f1"])
            self.val_f1s.append(val_metrics["f1"])

            print(f"[Epoch {epoch+1}] "f"Train Loss: {train_loss:.4f}, "f"Acc: {train_metrics['accuracy']:.4f}, "f"Prec: {train_metrics['precision']:.4f}, "f"Rec: {train_metrics['recall']:.4f}, "f"F1: {train_metrics['f1']:.4f}")
            print(f"[Epoch {epoch+1}] "f"Val Loss: {val_loss:.4f}, "f"Acc: {val_metrics['accuracy']:.4f}, "f"Prec: {val_metrics['precision']:.4f}, "f"Rec: {val_metrics['recall']:.4f}, "f"F1: {val_metrics['f1']:.4f}")

            # early stopping uses validation F1
            self.early_stopping(val_metrics["f1"], self.model)

            if self.early_stopping.early_stop:
                print("Early stopping triggered...")
                break
        
        # save loss plot
        self._save_loss_plot(save_path="results/loss_curve.png")
        self._save_f1_plot(save_path="results/f1_curve.png")

        # load best model
        self.early_stopping.load_best_model(self.model)
        print("Loaded best model.")



        # best model evaluation on train / val / test
        _, best_train_metrics = self.evaluate(data_loader=self.train_loader,loader_name="Training Evaluating...",)
        print("[Best Model Train] Classification Report:\n", best_train_metrics["classification_report"])
        print("[Best Model Train] Confusion Matrix:\n", best_train_metrics["confusion_matrix"])
        print()

        _, best_val_metrics = self.evaluate(data_loader=self.val_loader, loader_name="Validation Evaluating...",)
        print("[Best Model Validation] Classification Report:\n", best_val_metrics["classification_report"])
        print("[Best Model Validation] Confusion Matrix:\n", best_val_metrics["confusion_matrix"])
        print()

        _, test_metrics = self.evaluate(data_loader=self.test_loader, loader_name="Test Evaluating...",)
        print(f"[Test] Acc: {test_metrics['accuracy']:.4f}, "f"Prec: {test_metrics['precision']:.4f}, "f"Rec: {test_metrics['recall']:.4f}, "f"F1: {test_metrics['f1']:.4f}")
        print("Classification Report:\n", test_metrics["classification_report"])
        print("Confusion Matrix:\n", test_metrics["confusion_matrix"])
        print()

    def train_one_epoch(self):
        """One epoch training loop."""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        for wmap, labels in tqdm(self.train_loader, desc="Training", leave=False):
            x = wmap.to(self.device)
            labels = labels.to(self.device).long()

            _, outputs = self.model(x)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.compute_metrics(all_labels, all_preds)
        return avg_loss, metrics


    def evaluate(self, data_loader, loader_name: str):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for wmap, labels in tqdm(data_loader, desc=loader_name, leave=False):
                x = wmap.to(self.device)
                labels = labels.to(self.device).long()

                _, outputs = self.model(x)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self.compute_metrics(all_labels, all_preds)
        return avg_loss, metrics

    def compute_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "classification_report": report, "confusion_matrix": conf_matrix,}

    def _save_loss_plot(self, save_path="./results/loss_curve.png"):
        epochs = range(1, len(self.train_losses) + 1)
        best_idx = int(np.argmin(self.val_losses)) + 1
        plt.axvline(best_idx, linestyle="--", linewidth=1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Saved] Loss plot -> {save_path}")

    def _save_f1_plot(self, save_path="./results/f1_curve.png"):
        epochs = range(1, len(self.train_f1s) + 1)

        best_idx = int(np.argmax(self.val_f1s)) + 1  # 1-based epoch
        best_val_f1 = self.val_f1s[best_idx - 1]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_f1s, label="Train Macro-F1")
        plt.plot(epochs, self.val_f1s, label="Val Macro-F1")

        # best point (val)
        plt.axvline(best_idx, linestyle="--", linewidth=1)
        plt.scatter([best_idx], [best_val_f1], zorder=5, label=f"Best Val F1 (Ep {best_idx})")
        plt.annotate(
            f"Ep {best_idx}\nVal {best_val_f1:.4f}",
            (best_idx, best_val_f1),
            textcoords="offset points",
            xytext=(8, -12),
            ha="left",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("Train/Val Macro-F1 Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Saved] F1 plot -> {save_path}")