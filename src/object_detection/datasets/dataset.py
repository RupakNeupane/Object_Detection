import os
from PIL import Image

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


from src.object_detection.utils.utils import load_pickle

class OpenImageDataset(Dataset):

    def __init__(self, image_paths, csv_path):
        super().__init__()
        self.image_paths = image_paths
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.unique_images = self.df['ImageID'].unique()

    def __len__(self):
        return len(self.unique_images)
    
    def __getitem__(self, index):
        image_id = self.unique_images[index]
        image_full_path = os.path.join(os.getcwd(), self.image_paths, image_id + ".jpg")
        image = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        df = self.df.loc[self.df['ImageID'] == image_id]

        bboxes = df[['XMin', 'YMin', 'XMax', 'YMax']].values
        bboxes = (bboxes * np.array([w, h, w, h])).astype(np.uint16)

        classes = df['LabelName'].values
        return image, bboxes, classes, image_full_path
    
    

def preprocess(crop_img):
    crop_img = torch.tensor(crop_img).permute(2, 0, 1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_img = normalize(crop_img)
    return crop_img.float()


class RCNNDataset(Dataset):
    def __init__(self, fpaths, rois, gtbbs, labels, deltas, ious, label2target):
        super().__init__()
        self.fpaths = fpaths
        self.rois = rois
        self.gtbbs = gtbbs
        self.labels = labels
        self.deltas = deltas
        self.ious = ious
        self.label2target = label2target
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        fpath = self.fpaths[index]
        image = cv2.imread(fpath, cv2.IMREAD_COLOR)[..., ::-1]  # Convert BGR to RGB
        H, W, _ = image.shape

        gtbbs = self.gtbbs[index]

        rois = self.rois[index]
        bbs = (rois * np.array([W, H, W, H])).astype(np.uint8)  # Convert bounding boxes to integers

        # bbs is required because the Selective Search algorithm may return bounding boxes outside the image
        bbs = np.clip(bbs, 0, [W, H, W, H])

        
        crops = [image[y:Y, x:X] for x, y, X, Y in bbs]
        labels = self.labels[index]
        deltas = self.deltas[index]
        
        return image, gtbbs, bbs, crops, labels, deltas, fpath

    def collate_fn(self, batch):
        inputs, output_labels, output_deltas = [], [], []
        for i in range(len(batch)):
            image, gtbbs, bbs, crops, labels, deltas, fpath = batch[i]
            
            # Resize valid crops and preprocess them
            crops = [cv2.resize(crop,(224, 224)) for crop in crops ]
            crops = [preprocess(crop/255.0)[None] for crop in crops]
            
            inputs.extend(crops)
            output_labels.extend([self.label2target[label] for label in labels])
            output_deltas.extend(deltas)

    # yo tala ko 3 ota line le garda train garna lai feasible banako ho
        inputs = torch.cat(inputs).to(self.device)
        output_labels = torch.tensor(output_labels).long().to(self.device)
        output_deltas = torch.tensor(output_deltas).float().to(self.device)

        return inputs, output_labels, output_deltas


if __name__ == "__main__":
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

    train_dataset = RCNNDataset(FULL_PATHS, ROIS, GTBBS, CLSS, DELTAS, IOUS,label2target)
    
    img,gtbbs,bbs,crops,labels,deltas,fpath=train_dataset[0]
    print(img.shape)
    print(gtbbs.shape)
    print(bbs.shape)
    print(len(crops))
    print(len(deltas))
    print(len(labels))
    print(fpath)
