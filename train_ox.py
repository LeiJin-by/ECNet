import argparse
import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)

from model import ECNet, ECNet_model
from train import ImprovedECNetDataset
from utils.feature_engineering_ox import ECNet_ox_fea


def featurization_ox(data, config_path=None):
    formulas = data['composition'].values
    oxidation_states = data['oxidation_states'].values
    return ECNet_ox_fea(formulas, oxidation_states, config_path=config_path)


def build_models(name, j, save_model):
    return ECNet(name, j, save_model)


def load_model(name, j, device='cpu'):
    model_path = 'models/ECNet' + '_' + name + '_' + str(j) + '.pth'
    state_dict = torch.load(model_path, map_location=device)
    model = ECNet_model()
    model.load_state_dict(state_dict)
    return model


def train_ensemble_ox(data, weight, name, n_fold, device, lr, criterion, writer, epoch, folds=5, fold_seed=123, save_model=True, config_path=None, batch_size=32):
    y = data['target'].values
    index = data['materials-id'].values
    kfolds = KFold(n_splits=folds, shuffle=True, random_state=fold_seed)

    for j, (train, val) in enumerate(kfolds.split(index)):
        if j != n_fold:
            continue

        train_cv_X = featurization_ox(data.iloc[train], config_path=config_path)
        val_cv_X = featurization_ox(data.iloc[val], config_path=config_path)
        train_cv_weight = weight[train]

        model = build_models(name, j, save_model=save_model)
        train_cv_y = y[train]
        val_cv_y = y[val]

        print('/n======================Train ECNet-Ox========================\n')
        train_dataset = ImprovedECNetDataset(train_cv_X, train_cv_y, train_cv_weight)
        val_dataset = ImprovedECNetDataset(val_cv_X, val_cv_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model.trainer(device, train_loader, val_loader, lr=lr, criterion=criterion, writer=writer, epochs=epoch)
        del model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()


def get_train_data_ox(data, weight, name, device, lr, criterion, epoch=100, folds=5, fold_seed=123, save_model=True, config_path=None, batch_size=32):
    writer = SummaryWriter('./log/' + name)
    for i in range(folds):
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            f"--------------Train ECNet-Ox on fold {i + 1}--------------\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        )
        train_ensemble_ox(
            data,
            weight,
            name,
            i,
            device,
            lr,
            criterion,
            writer,
            epoch,
            folds=folds,
            fold_seed=fold_seed,
            save_model=save_model,
            config_path=config_path,
            batch_size=batch_size,
        )


def predict_ensemble_ox(name, j, data, device='cuda:0', config_path=None):
    features = featurization_ox(data, config_path=config_path)
    model = load_model(name, j, device=device)
    batchsize_0 = 8

    with torch.no_grad():
        model.to(device)
        model.eval()
        n_samples = len(features['element_ids'])
        dummy_labels = np.zeros(n_samples, dtype=np.float32)
        dataset = ImprovedECNetDataset(features, dummy_labels)
        loader = DataLoader(dataset, batch_size=batchsize_0, shuffle=False)

        predictions = []
        for batch in loader:
            features_batch = {k: v.to(device) for k, v in batch[0].items()}
            pred = model(features_batch)
            predictions.append(pred.cpu())

        y_pred = torch.cat(predictions, dim=0).numpy()
    return y_pred.reshape(-1, 1)


def y_to_01(y):
    return np.array([1 if value > 0.5 else 0 for value in y])


def performance(pre_test_y_prob, test_y):
    test_y = test_y.astype(int)
    pre_test_y = y_to_01(pre_test_y_prob)
    cm = confusion_matrix(test_y, pre_test_y, labels=[0, 1])

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    accuracy = accuracy_score(test_y, pre_test_y)
    precision_curve, recall_curve, _ = precision_recall_curve(test_y, pre_test_y_prob)
    aupr = auc(recall_curve, precision_curve)
    max_f1 = np.nanmax(2 * (precision_curve * recall_curve) / (precision_curve + recall_curve))
    precision = precision_score(test_y, pre_test_y, zero_division=0)
    recall = recall_score(test_y, pre_test_y)
    f1 = f1_score(test_y, pre_test_y)
    fnr = confusion_matrix(test_y, pre_test_y, normalize='pred')[1][0]
    auc_score = roc_auc_score(test_y, pre_test_y_prob)

    return accuracy, precision, recall, f1, fnr, auc_score, aupr, max_f1, tp, fp, tn, fn


def evaluate_ox(name, data, folds=5, device='cuda:0', config_path=None):
    pre_test_y = []
    for i in range(folds):
        pre_test_y_i = predict_ensemble_ox(name, i, data, device=device, config_path=config_path)
        pre_test_y.append(pre_test_y_i.ravel())

    pre_test = np.mean(pre_test_y, axis=0)
    target_y = data['target'].values
    return pre_test, performance(pre_test, target_y)


def main():
    parser = argparse.ArgumentParser(description="Training script for oxidation-state-aware ECNet.")
    parser.add_argument("--path", type=str, default='data/datasets/MP_ox_integer_valid.csv')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--train_data_used", type=float, default=1.0)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model", type=int, default=1)
    parser.add_argument("--performance_test", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split_seed", type=int, default=123)
    parser.add_argument("--subsample_seed", type=int, default=2)
    parser.add_argument("--fold_seed", type=int, default=123)
    parser.add_argument("--config_path", type=str, default='utils/elec_config_oxidation_state.csv')
    args = parser.parse_args()

    data = pd.read_csv(args.path)
    if 'oxidation_states' not in data.columns:
        raise ValueError("ECNet-Ox training requires an oxidation_states column.")

    criterion = torch.nn.BCELoss(reduction='sum')
    train_X, test_X, _, _ = train_test_split(
        data,
        data,
        test_size=0.1,
        random_state=args.split_seed,
    )
    if args.train_data_used < 1:
        train_X, _, _, _ = train_test_split(
            train_X,
            train_X,
            train_size=args.train_data_used,
            random_state=args.subsample_seed,
        )

    if args.train:
        weight = np.ones(len(train_X)) / len(train_X)
        get_train_data_ox(
            train_X,
            weight,
            args.name,
            args.device,
            args.lr,
            criterion,
            epoch=args.epochs,
            folds=args.folds,
            fold_seed=args.fold_seed,
            save_model=bool(args.save_model),
            config_path=args.config_path,
            batch_size=args.batch_size,
        )

    if args.performance_test:
        pre_test, perf = evaluate_ox(
            args.name,
            test_X,
            folds=args.folds,
            device=args.device,
            config_path=args.config_path,
        )
        accuracy, precision, recall, f1, fnr, auc_score, aupr, max_f1, tp, fp, tn, fn = perf
        print(f"""
        Performance Metrics:
        ====================
        Accuracy: {accuracy}
        Precision: {precision}
        Recall: {recall}
        F1 Score: {f1}
        False Negative Rate (FNR): {fnr}
        AUC Score: {auc_score}
        AUPR: {aupr}
        Max F1: {max_f1}

        Confusion Matrix Counts:
        ========================
        True Positive (TP):  {tp}
        False Positive (FP): {fp}
        False Negative (FN): {fn}
        True Negative (TN):  {tn}

        Total Samples: {tp + fp + tn + fn}
        """)


if __name__ == '__main__':
    main()
