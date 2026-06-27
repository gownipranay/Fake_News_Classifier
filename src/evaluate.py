"""
Evaluation utilities — metrics, plots, and reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
)


def plot_training_history(history, save_path: str = None):
    """Plot accuracy, loss, and AUC curves from Keras History."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    metrics_pairs = [
        ('accuracy', 'val_accuracy', 'Accuracy'),
        ('loss',     'val_loss',     'Loss'),
        ('auc',      'val_auc',      'AUC-ROC'),
    ]

    for ax, (train_m, val_m, title) in zip(axes, metrics_pairs):
        ax.plot(history.history.get(train_m, []), label='Train', linewidth=2)
        ax.plot(history.history.get(val_m, []),   label='Validation', linewidth=2, linestyle='--')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['FAKE', 'REAL'],
        yticklabels=['FAKE', 'REAL'],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true, y_probs, save_path: str = None):
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, save_path: str = None):
    """Precision-Recall curve with average precision annotation."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='steelblue', lw=2, label=f'AP = {ap:.4f}')
    ax.fill_between(recall, precision, alpha=0.1, color='steelblue')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def full_evaluation(model, X_test, y_test, threshold: float = 0.5):
    """
    Run full evaluation suite and print a summary report.

    Returns:
        dict with accuracy, auc, f1, precision, recall.
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    y_probs = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_probs >= threshold).astype(int)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'], digits=4))

    roc_auc = plot_roc_curve(y_test, y_probs)
    plot_precision_recall_curve(y_test, y_probs)
    plot_confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'auc':       roc_auc,
        'f1':        f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
    }

    print("\nSummary Metrics:")
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.4f}")

    return metrics
