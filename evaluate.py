import torch
import numpy as np
from sklearn import metrics
# from tqdm.auto import trange
from fastprogress.fastprogress import master_bar, progress_bar
from config import config


def get_class_predictions(encoder, dataset):
    encoder.eval()
    y_true, y_pred = [], []
    print("Running inference...")
    with torch.no_grad():
#         for index in trange(len(dataset), dynamic_ncols=True):
        for index in progress_bar(range(len(dataset))):
            # image, _, _ = dataset.get(index, transform=False)
            image_tensor, class_label, impression = dataset[index]
            image_tensor = image_tensor.unsqueeze(0).to(config.device)
            logits, feature = encoder(image_tensor)

            pred = torch.sigmoid(logits).cpu().detach().numpy()[0]
            true = class_label.cpu().detach().numpy()
        #     print("Ground-truth class labels", true)
        #     print("Predicted class probs", pred.round(3))
            y_pred.append(pred)
            y_true.append(true)
    return np.array(y_true), np.array(y_pred)

def get_class_predictions_lateral(encoder, dataset):
    encoder.eval()
    y_true, y_pred = [], []
    print("Running inference...")
    with torch.no_grad():
#         for index in trange(len(dataset), dynamic_ncols=True):
        for index in progress_bar(range(len(dataset))):
            # image, _, _ = dataset.get(index, transform=False)
            image_tensor_frontal, image_tensor_lateral, class_label, impression = dataset[index]
            image_tensor_frontal = image_tensor_frontal.unsqueeze(0).to(config.device)
            image_tensor_lateral = image_tensor_lateral.unsqueeze(0).to(config.device)
            logits, feature = encoder(image_tensor_frontal, image_tensor_lateral)

            pred = torch.sigmoid(logits).cpu().detach().numpy()[0]
            true = class_label.cpu().detach().numpy()
        #     print("Ground-truth class labels", true)
        #     print("Predicted class probs", pred.round(3))
            y_pred.append(pred)
            y_true.append(true)
    return np.array(y_true), np.array(y_pred)

def evaluate_encoder_predictions(y_true, y_pred):
    y_pred = (y_pred >= 0.1).astype(float)
#     print(y_pred.sum(axis=0))
    recall = metrics.recall_score(y_true, y_pred, average=None)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    ap = metrics.average_precision_score(y_true, y_pred, average=None)
    return precision, recall, ap, ap.mean()