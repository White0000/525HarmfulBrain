import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef
)

def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    average="macro",
    multi_label=False,
    threshold=0.5,
    topk=1,
    compute_cohen_kappa=False,
    compute_mcc=False
):
    if multi_label:
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= threshold).astype(int)
        trues = targets.cpu().numpy()
        acc = np.mean((preds == trues).all(axis=1))
        prec = precision_score(trues, preds, average=average, zero_division=0)
        rec = recall_score(trues, preds, average=average, zero_division=0)
        f1 = f1_score(trues, preds, average=average, zero_division=0)
        cm = None
        auc_val = None
        kappa = None
        mcc = None
        if preds.shape[1] == 1:
            try:
                auc_val = roc_auc_score(trues, probs)
            except:
                auc_val = None
        else:
            try:
                auc_val = roc_auc_score(trues, probs, average=average, multi_class="ovr")
            except:
                auc_val = None
        if compute_cohen_kappa:
            try:
                kappa = cohen_kappa_score(trues.ravel(), preds.ravel())
            except:
                kappa = None
        if compute_mcc:
            try:
                mcc = matthews_corrcoef(trues.ravel(), preds.ravel())
            except:
                mcc = None
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "auc": auc_val,
            "cohen_kappa": kappa,
            "mcc": mcc
        }
    else:
        if topk > 1:
            top_vals, top_indices = torch.topk(outputs, k=topk, dim=1)
            preds = top_indices.cpu().numpy()
            trues = targets.cpu().numpy()
            acc_count = 0
            for i in range(preds.shape[0]):
                if trues[i] in preds[i]:
                    acc_count += 1
            acc = acc_count / preds.shape[0]
            prec = None
            rec = None
            f1 = None
            cm = None
            auc_val = None
            kappa = None
            mcc = None
        else:
            _, p = torch.max(outputs, dim=1)
            preds = p.cpu().numpy()
            trues = targets.cpu().numpy()
            acc = accuracy_score(trues, preds)
            prec = precision_score(trues, preds, average=average, zero_division=0)
            rec = recall_score(trues, preds, average=average, zero_division=0)
            f1 = f1_score(trues, preds, average=average, zero_division=0)
            cm = confusion_matrix(trues, preds)
            num_classes = outputs.shape[1]
            auc_val = None
            if num_classes == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                try:
                    auc_val = roc_auc_score(trues, probs)
                except:
                    auc_val = None
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                try:
                    auc_val = roc_auc_score(trues, probs, average=average, multi_class="ovr")
                except:
                    auc_val = None
            kappa = None
            mcc = None
            if compute_cohen_kappa:
                try:
                    kappa = cohen_kappa_score(trues, preds)
                except:
                    kappa = None
            if compute_mcc:
                try:
                    mcc = matthews_corrcoef(trues, preds)
                except:
                    mcc = None
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "auc": auc_val,
            "cohen_kappa": kappa,
            "mcc": mcc
        }

def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device="cpu",
    average="macro",
    multi_label=False,
    threshold=0.5,
    topk=1,
    compute_cohen_kappa=False,
    compute_mcc=False
):
    model.eval()
    model.to(device)
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            all_outputs.append(logits.cpu())
            all_targets.append(y_batch.cpu())
    outputs_cat = torch.cat(all_outputs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(
        outputs_cat,
        targets_cat,
        average=average,
        multi_label=multi_label,
        threshold=threshold,
        topk=topk,
        compute_cohen_kappa=compute_cohen_kappa,
        compute_mcc=compute_mcc
    )
    return metrics

def demo_evaluation():
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        def forward(self, x):
            return self.fc(x)
    x_data = torch.randn(100, 10)
    y_data = torch.randint(0, 3, (100,))
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = DummyModel()
    metrics = evaluate_model(
        model,
        loader,
        device="cpu",
        average="macro",
        multi_label=False,
        threshold=0.5,
        topk=1,
        compute_cohen_kappa=True,
        compute_mcc=True
    )
    print("Evaluation metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
        else:
            print(f"confusion_matrix:\n{v}")

if __name__ == "__main__":
    demo_evaluation()
