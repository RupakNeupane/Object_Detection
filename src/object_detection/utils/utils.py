import selectivesearch
import numpy as np
import pickle
from tqdm import tqdm
import torch

def extract_candidates(img):
    _, regions = selectivesearch.selective_search(img, scale=4, min_size=20)
    candidates = []
    img_area = np.prod(img.shape[:2])
    for region in regions:
        if region['rect'] in candidates:
            continue
        if region['size'] < 0.05*img_area:
            continue
        if region['size'] > img_area:
            continue 
        candidates.append(region['rect'])
    return candidates

def extract_iou(bbox1, bbox2, epsilon=1e-5):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    width = x2 - x1
    height = y2 - y1

    if width < 0 or height < 0:
        return 0
    
    intersection_area = width * height 
    area_1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area_1 + area_2 - intersection_area
    return intersection_area / (union_area + epsilon)

def save_pickle(var, path):
    with open(path, 'wb') as file:
        pickle.dump(var, file)
        
def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


# training 

def train_batch(model, optimizer, inputs, actual_labels, deltas):
    model.train()
    optimizer.zero_grad()

    # forward pass
    _labels, _deltas = model(inputs)
    total_loss, classification_loss, localization_loss = model.calculate_loss(_labels, _deltas, actual_labels, deltas)
    conf, pred_labels = _labels.max(-1)
    acc = pred_labels == actual_labels

    # backward pass
    total_loss.backward()
    optimizer.step()

    return _labels, _deltas, total_loss, classification_loss, localization_loss, acc

@torch.no_grad
def validate_batch(model, inputs, actual_labels, deltas):
    model.eval()
    _labels, _deltas = model(inputs)
    total_loss, classification_loss, localization_loss = model.calculate_loss(_labels, _deltas, actual_labels, deltas)

    conf, pred_labels = _labels.max(-1)
    acc = pred_labels == actual_labels

    return _labels, _deltas, total_loss, classification_loss, localization_loss, acc

def train(n_epochs,train_dataloader,test_dataloader,model,optimizer):
    train_history = {
        'total_loss': [],
        'detection_loss': [],
        'localization_loss': [],
        'accuracy': []
    }

    test_history = {
        'total_loss': [],
        'detection_loss': [],
        'localization_loss': [],
        'accuracy': []
    }

    for epoch in range(1, n_epochs + 1):
        epoch_train_total_loss = 0
        epoch_train_detection_loss = 0
        epoch_train_localization_loss = 0
        epoch_train_acc = []

        for inputs, labels, deltas in tqdm(train_dataloader, desc=f'Training {epoch} of {n_epochs}'):
            _inputs ,_deltas, total_loss , classification_loss, localization_loss, acc = train_batch(model,optimizer, inputs, labels, deltas)
            epoch_train_total_loss += total_loss.item()
            epoch_train_detection_loss += classification_loss.item()
            epoch_train_localization_loss += localization_loss.item()
            epoch_train_acc.extend(acc.tolist())
            
        epoch_train_total_loss /= len(train_dataloader)
        epoch_train_detection_loss /= len(train_dataloader)
        epoch_train_localization_loss /= len(train_dataloader)
        epoch_train_acc = sum(epoch_train_acc)  / len(epoch_train_acc)
            
        epoch_test_total_loss = 0
        epoch_test_detection_loss = 0
        epoch_test_localization_loss = 0
        epoch_test_acc = []

        for inputs, labels, deltas in tqdm(test_dataloader, desc=f'Testing '):
            _inputs ,_deltas, total_loss ,classification_loss, localization_loss, acc = validate_batch(model, inputs, labels, deltas)
            epoch_test_total_loss += total_loss.item()
            epoch_test_detection_loss += classification_loss.item()
            epoch_test_localization_loss += localization_loss.item()
            epoch_test_acc.extend(acc.tolist())
            
        epoch_test_total_loss /= len(test_dataloader)   
        epoch_test_detection_loss /= len(test_dataloader)
        epoch_test_localization_loss /= len(test_dataloader)
        epoch_test_acc = sum(epoch_test_acc) / len(test_dataloader)

        train_history.get('total_loss').append(epoch_train_total_loss)
        train_history.get('detection_loss').append(epoch_train_detection_loss)
        train_history.get('localization_loss').append(epoch_train_localization_loss)
        train_history.get('accuracy').append(epoch_train_acc)
        
        test_history.get('total_loss').append(epoch_test_total_loss)
        test_history.get('detection_loss').append(epoch_test_detection_loss)
        test_history.get('localization_loss').append(epoch_test_localization_loss)
        test_history.get('accuracy').append(epoch_test_acc)
        
        print(f'Epoch {epoch} of {n_epochs}, Training_loss: {epoch_train_total_loss}, Testing Detection Loss: {epoch_test_total_loss}, Testing Localization Loss: {epoch_test_localization_loss}, Testing Accuracy: {epoch_test_acc}')

        return train_history,test_history
