# inbuilt packages 
import os 
from PIL import Image
from tqdm import tqdm 
import json

# Datascience Packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# PyTorch Related Packages 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import files
from src.object_detection.config.config import configuration
from src.object_detection.datasets.dataset import RCNNDataset
from src.object_detection.model.model import get_model
from src.object_detection.model.model import get_backbone
from src.object_detection.utils.utils import load_pickle,save_pickle,selectivesearch,train

def main():
    saved_path = os.path.join(os.getcwd(), "dump", configuration.get('saved_path') )
    model_path = os.path.join(saved_path, "model.pth")
    hyperparam_path = os.path.join(saved_path, "hyperparam.json")
    train_path = os.path.join(saved_path,'train_history.json')
    test_path = os.path.join(saved_path,'test_history.json')
    # train_curve_path = os.path.join(saved_path, 'train_curve.png')


    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FULL_PATHS = load_pickle(r'data\full_paths.pkl')
    GTBBS = load_pickle(r'data\gtbbs.pkl')
    CLSS = load_pickle(r'data\clss.pkl')
    DELTAS = load_pickle(r'data\deltas.pkl')
    ROIS = load_pickle(r'data\rois.pkl')
    IOUS = load_pickle(r'data\ious.pkl')
    unique_labels = np.unique(np.array([c for clss in CLSS for c in clss]))
    target2label = {i:label for i, label in enumerate(unique_labels)}
    label2target = {label:i for i, label in enumerate(unique_labels)}
    
        
    n_train = len(FULL_PATHS) * 8//10
    train_dataset = RCNNDataset(FULL_PATHS[:n_train], ROIS[:n_train], GTBBS[:n_train], CLSS[:n_train], DELTAS[:n_train], IOUS[:n_train],label2target)
    test_dataset = RCNNDataset(FULL_PATHS[n_train:], ROIS[n_train:], GTBBS[n_train:], CLSS[n_train:], DELTAS[n_train:], IOUS[n_train:],label2target)

    train_dataloader = DataLoader(train_dataset, batch_size=configuration.get('batch_size'), shuffle=True, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=configuration.get('batch_size'), shuffle=False, collate_fn=test_dataset.collate_fn)

    model = get_model(get_backbone(device), 3,label2target).to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=configuration.get('learning_rate'))

    train_history,test_history = train(configuration.get('n_epoch'),train_dataloader,test_dataloader,model,optimizer)

    torch.save(model,model_path)

    with open(hyperparam_path,"w") as h_fp:
        json.dump(configuration,h_fp)

    with open(train_path,"w") as t_fp:
        json.dump(train_history,t_fp)

    with open(test_path,"w") as t_fp:
        json.dump(test_history,t_fp)


if __name__=="__main__":
    main()