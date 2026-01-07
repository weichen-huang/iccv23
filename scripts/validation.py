import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import json
import os
from datetime import datetime


@dataclass
class FoldMetrics:
    fold_id: int
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')
    best_epoch: int = 0

    def update(self, train_loss: float, val_loss: float, epoch: int):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def to_dict(self) -> dict:
        return {
            'fold_id': self.fold_id,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class PatientKFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data: List[dict], patient_key: str = 'ptid') -> List[Tuple[np.ndarray, np.ndarray]]:
        patient_ids = np.array([sample[patient_key] for sample in data])
        n_samples = len(data)
        indices = np.arange(n_samples)

        gkf = GroupKFold(n_splits=self.n_splits)

        folds = []
        for train_idx, val_idx in gkf.split(indices, groups=patient_ids):
            folds.append((train_idx, val_idx))

        return folds

    def split_stratified(self, data: List[dict], patient_key: str = 'ptid',
                         label_key: str = 'dx') -> List[Tuple[np.ndarray, np.ndarray]]:
        patient_ids = np.array([sample[patient_key] for sample in data])

        labels = []
        for sample in data:
            dx = sample[label_key]
            if hasattr(dx, '__len__') and len(dx) > 1:
                labels.append(np.argmax(dx))
            else:
                labels.append(int(dx))
        labels = np.array(labels)

        n_samples = len(data)
        indices = np.arange(n_samples)

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                     random_state=self.random_state)

        folds = []
        for train_idx, val_idx in sgkf.split(indices, labels, groups=patient_ids):
            folds.append((train_idx, val_idx))

        return folds


class MetricsTracker:
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self.fold_metrics: List[FoldMetrics] = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def add_fold(self, fold_id: int) -> FoldMetrics:
        metrics = FoldMetrics(fold_id=fold_id)
        self.fold_metrics.append(metrics)
        return metrics

    def get_summary(self) -> Dict:
        if not self.fold_metrics:
            return {}

        best_val_losses = [m.best_val_loss for m in self.fold_metrics]
        best_epochs = [m.best_epoch for m in self.fold_metrics]

        return {
            'n_folds': len(self.fold_metrics),
            'mean_best_val_loss': float(np.mean(best_val_losses)),
            'std_best_val_loss': float(np.std(best_val_losses)),
            'min_best_val_loss': float(np.min(best_val_losses)),
            'max_best_val_loss': float(np.max(best_val_losses)),
            'mean_best_epoch': float(np.mean(best_epochs)),
            'per_fold': [m.to_dict() for m in self.fold_metrics]
        }

    def save(self, filename: Optional[str] = None):
        os.makedirs(self.output_dir, exist_ok=True)

        if filename is None:
            filename = f'cv_results_{self.timestamp}.json'

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

        return filepath

    def print_summary(self):
        summary = self.get_summary()

        print("\n" + "=" * 50)
        print("Cross-Validation Results")
        print("=" * 50)
        print(f"Number of folds: {summary['n_folds']}")
        print(f"Mean best validation loss: {summary['mean_best_val_loss']:.4f} ± {summary['std_best_val_loss']:.4f}")
        print(f"Best fold loss: {summary['min_best_val_loss']:.4f}")
        print(f"Worst fold loss: {summary['max_best_val_loss']:.4f}")
        print(f"Mean best epoch: {summary['mean_best_epoch']:.1f}")
        print("\nPer-fold results:")
        for fold_data in summary['per_fold']:
            print(f"  Fold {fold_data['fold_id']}: best_loss={fold_data['best_val_loss']:.4f} at epoch {fold_data['best_epoch']}")
        print("=" * 50 + "\n")


class CrossValidator:
    def __init__(
        self,
        n_folds: int = 5,
        early_stopping_patience: int = 5,
        output_dir: str = 'checkpoints',
        stratified: bool = True,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = output_dir
        self.stratified = stratified
        self.random_state = random_state

        self.kfold = PatientKFold(n_splits=n_folds, random_state=random_state)
        self.metrics_tracker = MetricsTracker(output_dir=output_dir)

    def get_folds(self, data: List[dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.stratified:
            return self.kfold.split_stratified(data)
        else:
            return self.kfold.split(data)

    def run(
        self,
        data: List[dict],
        model_factory: Callable,
        train_fn: Callable,
        val_fn: Callable,
        n_epochs: int = 64,
        verbose: bool = True
    ) -> Dict:
        folds = self.get_folds(data)
        os.makedirs(self.output_dir, exist_ok=True)

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            if verbose:
                print(f"\n{'=' * 50}")
                print(f"Fold {fold_idx + 1}/{self.n_folds}")
                print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
                print('=' * 50)

            train_data = [data[i] for i in train_indices]
            val_data = [data[i] for i in val_indices]

            model = model_factory()

            fold_metrics = self.metrics_tracker.add_fold(fold_idx)
            early_stopping = EarlyStopping(patience=self.early_stopping_patience)

            best_model_state = None

            for epoch in range(n_epochs):
                train_loss = train_fn(model, train_data, epoch)
                val_loss = val_fn(model, val_data)

                fold_metrics.update(train_loss, val_loss, epoch)

                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                if val_loss <= fold_metrics.best_val_loss:
                    best_model_state = model.state_dict().copy()

                if early_stopping(val_loss):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if best_model_state is not None:
                checkpoint_path = os.path.join(self.output_dir, f'fold_{fold_idx}_best.pt')
                torch.save(best_model_state, checkpoint_path)
                if verbose:
                    print(f"Saved best model to {checkpoint_path}")

        results_path = self.metrics_tracker.save()
        if verbose:
            self.metrics_tracker.print_summary()
            print(f"Results saved to {results_path}")

        return self.metrics_tracker.get_summary()
