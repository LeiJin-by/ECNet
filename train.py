import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import auc,recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score, accuracy_score, precision_recall_curve
import pickle


from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)

from model import  ECNet, ECNet_model
from utils.feature_engineering import ECNet_fea
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def get_formulas(data):
    formulas = data['composition'].values
    y = data['target'].values
    return formulas, y


def featurization(formulas, index, feature_path = None, load_from_local=False):
    if not load_from_local:

        f = ECNet_fea(formulas)

    else:
        res = open(feature_path, 'rb')
        data = pickle.load(res)
        full_dict = data 
        f = {key: full_dict[key][index] for key in full_dict.keys()}


    return f

class ImprovedECNetDataset(torch.utils.data.Dataset):
  
    def __init__(self, features_dict, labels, weights=None):
        self.element_ids = torch.from_numpy(features_dict['element_ids']).long()
        self.atom_counts = torch.from_numpy(features_dict['atom_counts']).float()
        self.electron_configs = torch.from_numpy(features_dict['electron_configs']).float()
        self.masks = torch.from_numpy(features_dict['masks']).bool()
        self.labels = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1)
        
        if weights is not None:
            self.weights = torch.from_numpy(weights.astype(np.float32)).reshape(-1, 1)
        else:
            self.weights = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = {
            'element_ids': self.element_ids[idx],
            'atom_counts': self.atom_counts[idx],
            'electron_configs': self.electron_configs[idx],
            'masks': self.masks[idx]
        }
        if self.weights is not None:
            return features, self.labels[idx], self.weights[idx]
        else:
            return features, self.labels[idx]


def load_model(name, j):
    m_path = 'models/ECNet'+ '_' + name + '_' + str(j) + '.pth'
    state_dict = torch.load(m_path)
    m = ECNet_model()

    m.load_state_dict(state_dict)
    return m



def build_models(name, j, save_model):
    model= ECNet(name, j, save_model)
    return model

def train_ensemble(data, weight, name, n_fold, device, lr, criterion, writer, epoch, folds=10, random_seed_3=123, save_model=True, feature_path=None, load_from_local=False):
    formulas = data['composition'].values
    y = data['target'].values

    index = data['materials-id'].values

    train_for = index
    train_y = y

    kfolds = KFold(n_splits=folds, shuffle=True, random_state=random_seed_3)

    j = 0
    for train, val in kfolds.split(train_for):
        if j == n_fold:
            train_cv_for = train_for[train]
            val_cv_for = train_for[val]

            train_cv_X = featurization(formulas[train], train_cv_for, feature_path, load_from_local)
            val_cv_X = featurization(formulas[val], val_cv_for, feature_path, load_from_local)

            train_cv_weight = weight[train]

            model = build_models(name, j, save_model=save_model)

            train_cv_y = train_y[train]
            val_cv_y = train_y[val]


            batchsize_0 = 32
            print('/n'
                '======================Train ECNet========================\n')
            train_dataset = ImprovedECNetDataset(train_cv_X, train_cv_y,train_cv_weight) 
            val_dataset = ImprovedECNetDataset(val_cv_X, val_cv_y)
            train_loader = DataLoader(train_dataset, batch_size=batchsize_0, shuffle=True,drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batchsize_0, shuffle=False)
            model.trainer(device, train_loader, val_loader, lr=lr,criterion=criterion, writer=writer, epochs=epoch)
            torch.cuda.empty_cache()
            
        j = j + 1


def predict_ensemble(save_path, name, j, data, device='cuda:0', feature_path=None, load_from_local=False):
    formulas = data['composition'].values
    y = data['target'].values
    index = data['materials-id'].values
    features = featurization(formulas, index, feature_path, load_from_local)

    pre_y = []

    m = load_model(name, j)


    batchsize_0 = 8


        
    with torch.no_grad():
                m.to(device)
                m.eval()
                n_samples = len(features['element_ids'])
                dummy_labels = np.zeros(n_samples, dtype=np.float32)
                dataset = ImprovedECNetDataset(features, dummy_labels)
                loader = DataLoader(dataset, batch_size=batchsize_0, shuffle=False)
                
                predictions = []
                for batch in loader: 
                    features_batch = {k: v.to(device) for k, v in batch[0].items()}
                    pred = m(features_batch)
                    predictions.append(pred.cpu())
                
                y_pred = torch.cat(predictions, dim=0).numpy()
                pre_y.append(y_pred)


    pre_y = [n.reshape(-1,1) for n in pre_y]
    return pre_y



def get_train_data(data, weight, name, device, lr, criterion, log= True, epoch=100, folds=10,  random_seed_3=123, save_model=True, train=True, feature_path= None, load_from_local=False):
    if log:
        writer = SummaryWriter('./log/' + name)
    if train:
        for i in range(folds):
            print(
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                f"--------------Train on fold {i + 1}--------------\n"
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            )
            train_ensemble(data, weight, name, i, device, lr, criterion, writer, epoch, folds=folds, random_seed_3=random_seed_3, save_model=save_model, feature_path=feature_path, load_from_local=load_from_local)

    return

def y_to_01(y):
    new_y = []
    for i in range(len(y)):
        if y[i] > 0.5:
            new_y.append(1)
        else:
            new_y.append(0)
    return np.array(new_y)

def Performance( pre_test_y_prob, test_y):

    test_y = test_y.astype(int)
    pre_test_y = y_to_01(pre_test_y_prob)

    cm = confusion_matrix(test_y, pre_test_y, labels=[0, 1])
    

    tn = cm[0, 0] 
    fp = cm[0, 1]  
    fn = cm[1, 0]  
    tp = cm[1, 1] 
    
    
    accuracy = accuracy_score(test_y, pre_test_y)

    precision, recall, _ = precision_recall_curve(test_y, pre_test_y_prob)
    aupr = auc(recall, precision)
    max_f1 = max(2 * (precision * recall) / (precision + recall))


    precision = precision_score(test_y, pre_test_y, zero_division=0)
    recall = recall_score(test_y, pre_test_y)
    f1 = f1_score(test_y, pre_test_y)
    fnr = confusion_matrix(test_y, pre_test_y, normalize='pred')[1][0]
    auc_score = roc_auc_score(test_y, pre_test_y_prob)



    return accuracy,precision,recall,f1,fnr,auc_score, aupr,max_f1,tp,fp,tn,fn


def evaluate(name, data, folds=10):
    pre_test_y = []
    for i in range(folds):
        pre_test_y_i = predict_ensemble('models', name, i, data)
        pre_test_y_j = pre_test_y_i[0].ravel()
        pre_test_y.append(pre_test_y_j)

    pre_test = np.mean(pre_test_y, axis=0)


    target_y = data['target'].values

    performance = Performance(pre_test, target_y)
    return pre_test, performance


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script for the machine learning model")


    parser.add_argument("--path", type=str, default='data/datasets/demo_mp_data.csv',
                        help="Path to the dataset (default: data/datasets/demo_mp_data.csv")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model, default: 100")
    parser.add_argument("--train", type=int, default=1,
                        help="whether to train the model, 1: true, 0: false, default: 1")
    parser.add_argument("--name", type=str, help="Name of the experiment or model")
    parser.add_argument("--train_data_used", type=float, default=1.0,
                        help="Fraction of training data to be used")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device to run the training on, e.g., 'cuda:0' or 'cpu', default: 'cuda:0'")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds for training ECSG, default: 5")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--save_model", type=int, default=1,
                        help="Whether to save trained models , 1: true, 0: false, default: 1")
    parser.add_argument("--load_from_local", type=int, default=0,
                        help="Load features from local or generate features from scratch , 1: true, 0: false, default: 0")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="Path to processed features, default: None")
    parser.add_argument("--prediction_model", type=int, default=0,
                        help="Train a model for predicting or testing , 1: true, 0: false, default: 0")
    parser.add_argument("--performance_test", type=bool, default=True,
                        help="Whether to test the performance of trained model , 1: true, 0: false, default: 1")

    args = parser.parse_args()
    device = args.device
    print(device)

    """tasks type"""
    train = args.train

    save_model = args.save_model   
    load_from_local = args.load_from_local  
    prediction_model = args.prediction_model  
    performance_test = args.performance_test



    """hyperparameters"""
    criterion = torch.nn.BCELoss(reduction='sum')
    lr = args.lr
    epoch = args.epochs
    name = args.name
    folds = args.folds
    train_data_used = args.train_data_used


    write = SummaryWriter('log/'+ name)
    '''data_path'''
    path = args.path
    data = pd.read_csv(path)

    """select seed"""
    random_seed_1 = 123
    random_seed_2 = 2
    random_seed_3 = 123


    train_X, test_X, _, _ = train_test_split(data, data, test_size=0.1, random_state=random_seed_1)
    if train_data_used < 1:
        train_X, U_X, _, _ = train_test_split(train_X, train_X, train_size=train_data_used, random_state=random_seed_2)

    if prediction_model:
        train_X = data
        test_X = pd.read_csv('data/datasets/test_X.csv')

    if train:
        weight = np.ones(len(train_X)) / len(train_X)
        get_train_data(train_X, weight, name, device, lr, criterion, epoch=epoch, folds=folds,  random_seed_3=random_seed_3, save_model=save_model, train=True)

    if performance_test:
        pre_test, performance = evaluate(name, test_X, folds=folds)
        accuracy, precision, recall, f1, fnr, auc_score, aupr, max_f1,tp,fp,tn,fn = performance


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
