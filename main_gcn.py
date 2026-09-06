import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn, optim
from utils.utils import EarlyStopping, set_seed, resize_wafer_map
from utils.trainer import Trainer
from dataset.dataset import WaferDataset, AugmentedWaferDataset
from model.model_gcn import ViG_Classifier
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode




if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    imgsize = 128
    data_seed = 29
    print(f'--- data_seed: {data_seed} --- \n')

    # Data Loading and Preprocessing
    data_path = 'C:/Users/taehyeok/Desktop/gnn-classification/data/wm811k-wafer-map/with_label/wafer-map-with-label.pkl'
    df = pd.read_pickle(filepath_or_buffer=data_path)
    wmaps = df['waferMap'].copy()
    labels = df['failureNum'].copy()
    resized_wmaps = np.array([resize_wafer_map(w, size=(imgsize, imgsize)) for w in tqdm(wmaps)])

    # Data Splitting
    tr_val_data, test_data, tr_val_labels, test_labels = train_test_split(resized_wmaps, labels, test_size=0.1, random_state=data_seed, stratify=labels)
    tr_data, val_data, tr_labels, val_labels = train_test_split(tr_val_data, tr_val_labels, test_size=0.2, random_state=data_seed, stratify=tr_val_labels)

    augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation((-180, 180), interpolation=InterpolationMode.NEAREST),
    transforms.RandomAffine(degrees=0, 
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05),
                            interpolation=InterpolationMode.NEAREST,
                            fill=0)
    ])
    target_counts = {0:6970, 1:6970, 2:6970, 4:6970, 5:6970, 6:6970, 7:6970}
    train_dataset = AugmentedWaferDataset(tr_data, tr_labels, target_counts=target_counts, augment_transform=augment_transform)
  
    val_dataset = WaferDataset(val_data, val_labels)
    test_dataset = WaferDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # hpyerparameterss
    model = ViG_Classifier(patch_emb=256, num_blocks=2, hidden_features=512, n_classes=9, k=7, dilation=1, grid_size=8).to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs = 30
    os.makedirs("./checkpoints", exist_ok=True)
    model_save_path = "./checkpoints/seed_experiment/GCN/VIG_classification-GCN-k7_seed{}.pth".format(data_seed)

    
    early_stopping = EarlyStopping(patience=10, delta=0.0, path=model_save_path, verbose=True)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        criterion=criterion,
        epochs=epochs,
        optimizer=optimizer,
        early_stopping=early_stopping,
        device=device,
    )
    
    print(model)
    trainer.training()