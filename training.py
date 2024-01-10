import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import r2_score
from model import CustomDataset, Flip, Rotate90, Reflect, Identity
from torch.utils.tensorboard import SummaryWriter
from model import load_data, LearningMethod, RetNet, init_weights

# For reproducible results.
# See also -> https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(1)
torch.manual_seed(1)
random.seed(1)

# Requires installation with GPU support.
# See also -> https://pytorch.org/get-started/locally/
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training data.
X_train, y_train = load_data(
        'data/MOFs/batch_train',
        'data/MOFs/all_MOFs_screening_data.csv',
        'CO2_uptake_P0.15bar_T298K [mmol/g]',
        'MOFname',
        )

# Load validation data.
X_val, y_val = load_data(
        'data/MOFs/batch_val_test',
        'data/MOFs/all_MOFs_screening_data.csv',
        'CO2_uptake_P0.15bar_T298K [mmol/g]',
        'MOFname',
        size=5_000
        )

# Transformations for standardization + data augmentation.
standardization = transforms.Normalize(X_train.mean(), X_train.std())

augmentation = transforms.Compose([
    standardization,
    transforms.RandomChoice([Rotate90(), Flip(), Reflect(), Identity()]),
    ])

# Adding a channel dimension required for CNN.
X_train, X_val = [X.reshape(X.shape[0], 1, *X.shape[1:]) for X in [X_train, X_val]]

# Create the dataloaders.
train_loader = DataLoader(
        CustomDataset(X=X_train, y=y_train, transform_X=augmentation),
        batch_size=64, shuffle=True, pin_memory=True,
        )

val_loader = DataLoader(
        CustomDataset(X=X_val, y=y_val, transform_X=standardization),
        batch_size=512, pin_memory=True,
        )

# Define the architecture, loss and optimizer.
net = RetNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Define the learning rate scheduler.
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10,
    gamma=0.5, verbose=True
    )

# Initialize weights.
net.apply(lambda m: init_weights(m, a=0.01))

# Initialize bias of the last layer with E[y_train].
torch.nn.init.constant_(net.fc[-1].bias, y_train.mean())

model = LearningMethod(net, optimizer, criterion) 
print(net)
model_name = 'RetNet'

# Use Tensorboard. Needs to be fixed!
# See also -> https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
#writer = SummaryWriter(log_dir='experiments/')

model.train(
    train_loader=train_loader, val_loader=val_loader,
    metric=r2_score, epochs=1, scheduler=scheduler,
    device=device, verbose=True, #tb_writer=writer,
    )

# Calculate R^2 on the whole validation set.
predictions = [model.predict(x.to(device)) for x, _ in val_loader]

y_pred = torch.concatenate(predictions)
y_true = torch.tensor(y_val).reshape(len(y_val), -1).to(device)

print(f'R2 for validation set: {r2_score(input=y_pred, target=y_true)}')

# Save the trained model.
# See also -> https://pytorch.org/tutorials/beginner/saving_loading_models.html
#torch.save(model, f'{model_name}.pt')
