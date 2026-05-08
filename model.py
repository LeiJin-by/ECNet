import pandas as pd
import numpy as np
from time import time


import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout
import torch.nn.functional as F
import sys
from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)



class ECNet_model(nn.Module):
    def __init__(self,
                 electron_config_dim=137,
                 embedding_dim=128,
                 num_elements=118,
                 max_elements=15,
                 dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fixed_element_proj = nn.Linear(num_elements + 1, embedding_dim)
        self.electron_projection = nn.Linear(electron_config_dim, embedding_dim)
        self.count_projection = nn.Linear(8, embedding_dim)
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, features):
        eid = features['element_ids']
        counts = features['atom_counts']
        elec = features['electron_configs']
        masks = features['masks']

        one_hot = torch.nn.functional.one_hot(eid, num_classes=119).float()  
        elem_emb = self.fixed_element_proj(one_hot)  
        elec_emb = self.electron_projection(elec)
        cnt_emb = self.count_projection(counts)
        fused = self.feature_fusion(torch.cat([elem_emb, elec_emb, cnt_emb], dim=-1))
        
        x = fused.transpose(1, 2) 
        x = self.conv(x)
        B, C, Lp = x.size()
        m = (~masks).float() 
        m_ds = F.interpolate(m.unsqueeze(1), size=Lp, mode='nearest').squeeze(1) 
        m_ds = (m_ds > 0.5).float()
        denom = m_ds.sum(dim=1, keepdim=True).clamp_min(1.0) 
        x = (x * m_ds.unsqueeze(1)).sum(dim=2) / denom 
        return self.head(x)



def get_right_count(output, target):
    count = 0
    for i in range(len(output)):
        if output[i] > 0.5:
            output[i] = 1
        else:
            output[i] = 0

        if output[i] == target[i]:
            count += 1

    return count



class ECNet():
    def __init__(self, name, number, save_model=True):
        self.model_name = 'ECNet' + '_' + name + '_' + str(number)
        self.save_path = './models/' + self.model_name
        self.save_model = save_model
    def build_model(self):
        self.model = ECNet_model()
        return self.model
    def train(self, device, train_loader, lr, criterion, epoch):
        size = len(train_loader.dataset)

        self.device = device
        self.model.to(self.device)
        self.model.train()
        train_loss = 0
        right_count = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for batch, (input, target, weight) in enumerate(train_loader):
            input_features = {k: v.to(self.device) for k, v in input.items()}
            target = target.to(self.device)
            weight = weight.to(self.device)

            output = self.model(input_features)
            s_loss = criterion(output, target)
            loss = (s_loss * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right_count += get_right_count(output, target)
            train_loss += loss * len(target)
            del loss

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = right_count / len(train_loader.dataset)
        
        return train_loss, train_acc

    def valuate(self, test_loader, criterion, min_loss, max_acc):
        self.model.eval()
        test_loss, right_count = 0, 0
        with torch.no_grad():
            for batch, (input, target) in enumerate(test_loader):
                input_features = {k: v.to(self.device) for k, v in input.items()}
                target = target.to(self.device)

                output = self.model(input_features)
                test_loss += criterion(output, target).sum()
                right_count += get_right_count(output, target)

            test_loss /= len(test_loader.dataset)
            acc = right_count / len(test_loader.dataset)
            
            if test_loss < min_loss:
                min_loss = test_loss

            if acc > max_acc:
                max_acc = acc

        return test_loss, min_loss, acc, max_acc
    
    def trainer(self, device, train_loader, test_loader, lr, criterion, writer, epochs):
        min_loss = 1e20
        max_acc = -100
        self.build_model()
        for i in range(epochs):
            start_time = time()
            train_loss, train_acc = self.train(device, train_loader, lr, criterion, i)
            end_time = time()
            test_loss, min_loss, acc, max_acc = self.valuate(test_loader, criterion, min_loss, max_acc)


            writer.add_scalar('loss/train_' + self.model_name, train_loss, i)
            writer.add_scalar('acc/train_' + self.model_name, train_acc, i)
            writer.add_scalar('loss/test_' + self.model_name, test_loss, i)
            writer.add_scalar('acc/test_' + self.model_name, acc, i)
            dur = end_time - start_time

            print(f'epoch {i}, time:{dur:>4f}         ==============================================')
            print(f'train_loss is {train_loss:>5f}, test_loss is {test_loss:>5f}, acc is {acc:>3f}, max acc is {max_acc:>3f}')
            if test_loss == min_loss:
                print(f'test_loss < min_loss, save {self.model_name}')
            if self.save_model:
                torch.save(self.model.state_dict(), self.save_path + '.pth')

    def predict(self, X):
        self.model.eval()
        X_features = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(self.device) 
                     for k, v in X.items()}
        with torch.no_grad():
            y = self.model(X_features)
        return y


