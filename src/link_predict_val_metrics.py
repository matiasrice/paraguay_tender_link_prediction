from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import tqdm
import torch.nn.functional as F
import numpy as np
import random
import torch

def validation_metrics(model, sampled_data, val_loader, set_seed=42, show_results=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(set_seed)
    np.random.seed(set_seed)
    torch.manual_seed(set_seed)

    preds = []
    ground_truths = []

    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["institution", "rates", "supplier"].edge_label)
    
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    # Compute ROC AUC
    auc = roc_auc_score(ground_truth, pred)

    # Compute Accuracy
    accuracy = accuracy_score(ground_truth, (pred > 0.5).astype(int))

    # Compute Precision
    precision = precision_score(ground_truth, (pred > 0.5).astype(int))

    # Compute Recall
    recall = recall_score(ground_truth, (pred > 0.5).astype(int))

    if show_results:
        print()
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")

    return {'roc_auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall}

import numpy as np
from sklearn.metrics import precision_score
import random

def precision_at_k(model, val_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    preds = []
    ground_truths = []

    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["institution", "rates", "supplier"].edge_label)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    total_k = len(ground_truth)
    true_idx = np.where((preds>0.5)==True)[0]
    preds = preds[true_idx]
    ground_truth = ground_truth[true_idx]

    precision_values = []

    for i in range(1, len(ground_truth)+1):
        # Efficiently get the indices of the top k predictions
        top_k_indices = np.argsort(-preds)[:i]

        # Get the corresponding ground truth labels for the top k predictions
        top_k_ground_truth = ground_truth[top_k_indices]

        # Compute precision at k
        precision_k = precision_score(top_k_ground_truth, (preds[top_k_indices] > 0.5).astype(int))
        precision_values.append(precision_k)

    return precision_values, total_k