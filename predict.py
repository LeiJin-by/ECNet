import argparse
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from os.path import dirname, abspath

path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)

from model import ECNet_model
from utils.feature_engineering import ECNet_fea




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





def predict_single_fold(name, j, data, device='cuda:0', feature_path=None, load_from_local=False):
    formulas = data['composition'].values
    #y = data['target'].values
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




def predict_avg(name, data, folds=10):
    pre_test_y = []
    for i in range(folds):
        pre_test_y_i = predict_single_fold( name, i, data)
        pre_test_y_j = pre_test_y_i[0].ravel()
        pre_test_y.append(pre_test_y_j)

    pre_test = np.mean(pre_test_y, axis=0)


    return pre_test




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/datasets/demo_mp_data.csv')
    parser.add_argument("--name", type=str, help="Name of the experiment or model")
    parser.add_argument("--load_from_local", type=int, default=0,
                        help="Load features from local or generate features from scratch , 1: true, 0: false, default: 0")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="Path to processed features, default: None")


    args = parser.parse_args()
    name = args.name

    '''data_path'''
    path = args.path
    predict_data = pd.read_csv(path)

    results = predict_avg(name, predict_data, folds=5)
    predict_data = predict_data.rename(columns={'target': 'pre_y'})
    predict_data['pre_y'] = results
    
    results = [True if results[n] > 0.5 else False for n in range(len(results))]
    predict_data['pre_y_01'] = results
    
    save_path = 'results/' + name + '_predict_results.csv'
    predict_data.to_csv(save_path, index=False)
    print(f'Prediction results saved in {save_path}')



