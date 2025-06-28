import numpy as np
import imageio.v3 as iio
import scipy.ndimage
import math
from sklearn.decomposition import PCA
from sklearn import svm
import os
import scipy.signal
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import cv2

def normalize(l):
    l = l.astype(float)
    return (l - l.min())/(l.max() - l.min())

def luminance(l):
    if len(l.shape) > 2:
        l = 0.2126 * l[:, :, 0] + 0.7152 * l[:, :, 1] + 0.0722 * l[:, :, 2]
        l = l.astype(np.uint8)
    return l

def feat_from_image(img_path, model="hog"):
    if model == "hog":
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        winSize = (224,224)
        blockSize = (112,112)
        blockStride = (56,56)
        cellSize = (56,56)
        nbins = 9
            
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog_feats = hog.compute(img)
        return list(hog_feats.flatten())
    elif model == "color":
        img = iio.imread(img_path)
        r, _ = np.histogram(img[..., 0], bins=range(16))
        g, _ = np.histogram(img[..., 1], bins=range(16))
        b, _ = np.histogram(img[..., 2], bins=range(16))
        
        return np.concatenate([r, g, b]).astype(float)/(img.shape[0]*img.shape[1])


def make_prediction(dataset_path, dataset_csv, labels, desc_type="hog"):
    print(f"Testando modelo {desc_type}")
    train_feats = []
    train_labels = []

    print("Carregando dados de treino...")
    # Carregando dados de treino
    with open(dataset_csv, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            img_file, phase, label = line.strip().split(",")
            label = True if label == "True" else False

            if phase != "train":
                continue

            img_path = f"{dataset_path}/{phase}/{labels[label]}/{img_file}"
            img_feat = feat_from_image(img_path, desc_type)

            train_feats.append(img_feat)
            train_labels.append(label)

    print("Treinando o modelo...")
    model = svm.SVC(kernel="rbf")
    model.fit(train_feats, train_labels)

    print("Carregando dados de teste...")
    test_feats = []
    test_labels = []
    
    with open(dataset_csv, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            img_file, phase, label = line.strip().split(",")
            label = True if label == "True" else False

            if phase == "train":
                continue

            img_path = f"{dataset_path}/{phase}/{labels[label]}/{img_file}"
            img_feat = feat_from_image(img_path, desc_type)

            test_feats.append(img_feat)
            test_labels.append(label)

    preds = model.predict(test_feats)

    return accuracy_score(test_labels, preds)



if __name__ == "__main__":
    dataset_path = "./dataset/the_wildfire_dataset_2n_version"
    dataset_csv = "images.csv"
    labels = ["nofire", "fire"]

    print(make_prediction(dataset_path, dataset_csv, labels, desc_type="hog"))
    print(make_prediction(dataset_path, dataset_csv, labels, desc_type="color"))