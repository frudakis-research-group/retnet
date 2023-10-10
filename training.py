import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score
from model import CustomDataset, Flip, Roll, Rotate90, Reflect, Identity
from torch.utils.tensorboard import SummaryWriter
from model import load_data, LearningMethod, DummyNet, VoNet, init_weights


# For reproducible results
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

# Elements...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_size = 55_871

# Loading data
X_train, y_train = load_data(
        #'data/batches/batch_train',
        'data/COFs/batch_train',
        #'data/all_MOFs_screening_data.csv',
        'data/COFs/COFs_low_pressure.csv',
        #'CO2_uptake_P0.15bar_T298K [mmol/g]',
        'adsV_CH4_5.8b',
        #'MOFname',
        'COFname',
        #size=train_size
        )

X_val, y_val = load_data(
        #'data/batches/batch_val_test',
        'data/COFs/batch_val_test',
        #'data/all_MOFs_screening_data.csv',
        'data/COFs/COFs_low_pressure.csv',
        #'CO2_uptake_P0.15bar_T298K [mmol/g]',
        'adsV_CH4_5.8b',
        #'MOFname',
        'COFname',
        #size=5_000
        )

# Transformations for standardization/data augmentation
standardization = transforms.Normalize(X_train.mean(), X_train.std())

augmentation = transforms.Compose([
    standardization,
    transforms.RandomChoice([Rotate90(), Flip(), Reflect(), Identity()]),
    ])

# Adding a channel dimension required for CNN
X_train, X_val = [X.reshape(X.shape[0], 1, *X.shape[1:]) for X in [X_train, X_val]]

# Create the dataloaders
train_loader = DataLoader(
        CustomDataset(X=X_train, y=y_train, transform_X=augmentation),
        batch_size=64, shuffle=True, pin_memory=True
        )

val_loader = DataLoader(
        CustomDataset(X=X_val, y=y_val, transform_X=standardization),
        batch_size=512, pin_memory=True
        )

# Model architecture and optimization
net = VoNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10,
    gamma=0.5, verbose=True
    )

# Initialize weights and last bias with E[y_train]
net.apply(lambda m: init_weights(m, a=0.01))
torch.nn.init.constant_(net.fc[-1].bias, y_train.mean())

model = LearningMethod(net, optimizer, criterion) 

print(net)

model_name = f'VoNet'
#writer = SummaryWriter(log_dir=f'experiments/{model_name}')

model.train(
    train_loader=train_loader, val_loader=val_loader,
    metric=r2_score, epochs=50, scheduler=scheduler,
    cuda_device=device, verbose=True, #tb_writer=writer,
    )

#predictions = []
#for x, _ in val_loader:
#    y_pred = model.predict(x.to(device))
#    predictions.append(y_pred)
#
#y_pred = torch.concatenate([b for b in predictions])
#y_true = torch.tensor(y_val).reshape(len(y_val), -1).to(device)
#
#print(f'R2 for test set: {r2_score(y_pred, y_true)}')

#torch.save(model, f'saved_models/COFs/with_data_augmentation/{model_name}_{train_size}.pkl')

#with open(f'saved_models/COFs/with_data_augmentation/{model_name}_{train_size}.pkl', 'wb') as fhand:
#    pickle.dump(model, fhand)
