import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from itertools import cycle, combinations
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import r2_score


class RetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=12),
            nn.LeakyReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, bias=False),
            nn.BatchNorm3d(num_features=24),
            nn.LeakyReLU(),
            )

        self.max1 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=32, kernel_size=2, bias=False),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
            )

        self.max2 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.LeakyReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=120, kernel_size=2, bias=False),
            nn.BatchNorm3d(num_features=120),
            nn.LeakyReLU(),
            )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(3*3*3*120, 84),
            nn.BatchNorm1d(num_features=84),
            nn.LeakyReLU(),
            nn.Linear(84, 20),
            nn.BatchNorm1d(num_features=20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            )
    
    def forward(self, x):
         x = self.conv1(x)
         x = self.conv2(x)
         x = self.max1(x)
         x = self.conv3(x)
         x = self.max2(x)
         x = self.conv4(x)
         x = self.conv5(x)
         x = self.fc(x)
         
         return x


class VoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm3d(num_features=12),
            nn.LeakyReLU(),
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, bias=False),
            nn.BatchNorm3d(num_features=24),
            nn.LeakyReLU(),
            )

        self.max1 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=24, out_channels=32, kernel_size=2, bias=False),
            nn.BatchNorm3d(num_features=32),
            nn.LeakyReLU(),
            )

        self.max2 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=2, bias=False),
            nn.BatchNorm3d(num_features=64),
            nn.LeakyReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=120, kernel_size=2, bias=False),
            nn.BatchNorm3d(num_features=120),
            nn.LeakyReLU(),
            )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.3),
            nn.Linear(3*3*3*120, 84),
            nn.BatchNorm1d(num_features=84),
            nn.LeakyReLU(),
            nn.Linear(84, 20),
            nn.BatchNorm1d(num_features=20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            )
    
    def forward(self, x):
         x = self.conv1(x)
         x = self.conv2(x)
         x = self.max1(x)
         x = self.conv3(x)
         x = self.max2(x)
         x = self.conv4(x)
         x = self.conv5(x)
         x = self.fc(x)
         
         return x


class LearningMethod:
    def __init__(self, network, optimizer, criterion):
        self.net = network
        self.optimizer = optimizer
        self.criterion = criterion

    def train(
        self, train_loader, val_loader,
        val_loss_freq=15, epochs=1, scheduler=None,
        metric=r2_score, device=None, tb_writer=None, verbose=True,
        ):
        
        self.scheduler = scheduler
        self.val_loss_freq = val_loss_freq
        self.train_hist = []
        self.train_metric = []
        self.val_hist = []
        self.val_metric = []
        self.writer = tb_writer
        self.train_batch_size = train_loader.batch_size
        self.val_batch_size = val_loader.batch_size
        self.epochs = epochs

        val_loader = cycle(val_loader)

        # Training phase.
        counter = 0
        for e in range(epochs):

            if verbose:
                print(f'\nEpoch: {e}')

            for i, (X_train, y_train) in enumerate(train_loader):
                X_train, y_train = X_train.to(device), y_train.to(device)

                # Initialize zero gradients.
                self.optimizer.zero_grad()

                y_train_hat = self.net(X_train)
                train_loss = self.criterion(y_train_hat.ravel(), y_train)

                if (i % val_loss_freq == 0):
                    counter += val_loss_freq

                    X_val, y_val = next(val_loader)
                    X_val, y_val = X_val.to(device), y_val.to(device)

                    # Account for Dropout + BatchNorm.
                    yth = self.predict(X_train)

                    y_val_hat = self.predict(X_val)
                    val_loss = self.criterion(y_val_hat.ravel(), y_val)

                    train_metric = metric(yth.ravel(), y_train)
                    val_metric = metric(y_val_hat.ravel(), y_val)

                # Update the parameters.
                train_loss.backward()
                self.optimizer.step()

                # Print train and validation loss per `val_loss_freq` is.
                if verbose and (i % val_loss_freq == 0):
                    print(
                    f'{f"Iteration {counter}":<20} ->',
                    #f'{f"train_loss = {train_loss:.3f}":<22}',
                    f'{f"train_metric = {train_metric:.3f}":<22}',
                    #f'{f"val_loss = {val_loss:.3f}":>22}', sep=4*' '
                    f'{f"val_metric = {val_metric:.3f}":>22}', sep=4*' '
                    )

            self.train_hist.append(train_loss.item())
            self.train_metric.append(train_metric.item())
            self.val_hist.append(val_loss.item())
            self.val_metric.append(val_metric.item())

            if tb_writer:
                self.writer.add_scalars(
                    'learning_curve',
                    #{'train': train_loss, 'val': val_loss},
                    {'train': train_metric, 'val': val_metric},
                    e
                    )
                self.writer.add_scalar('Metric/train', train_metric, e)
                self.writer.add_scalar('Metric/val', val_metric, e)

                for name, value in self.net.named_parameters():
                    self.writer.add_histogram(f'Values/{name}', value, e) 
                    self.writer.add_histogram(f'Gradients/{name}', value.grad,  e) 

            if scheduler:
                self.scheduler.step()

        if tb_writer:
            self.writer.flush()
            self.writer.close()

        print('\nTraining finished!')

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(X)
        self.net.train()

        return y_pred


class CustomDataset(Dataset):
    def __init__(self, X, y, transform_X=None, transform_y=None):
        self.transform_X = transform_X
        self.transform_y = transform_y
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample_x = torch.tensor(self.X[idx])
        sample_y = torch.tensor(self.y[idx])

        if self.transform_X:
            sample_x = self.transform_X(sample_x)
        if self.transform_y:
            sample_y = self.transform_y(sample_y)

        return sample_x, sample_y
            

class Rotate90:
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))
        self.n_choices = len(self.planes)

    def __call__(self, sample):
        plane = self.planes[np.random.choice(self.n_choices)]
        direction = np.random.choice([-1, 1])

        return torch.rot90(sample, k=direction, dims=plane)


class Flip:
    def __call__(self, sample):
        axis = np.random.choice([1, 2, 3])

        return torch.flip(sample, [axis])


class Reflect:
    def __init__(self):
        self.planes = list(combinations([1, 2, 3], 2))
        self.n_choices = len(self.planes)

    def __call__(self, sample):
        plane = self.planes[np.random.choice(self.n_choices)]

        return torch.transpose(sample, *plane)


class Roll:
    def __call__(self, sample):
        axis = np.random.choice([1, 2, 3])
        shift = np.random.choice([1, 2, 4, 6, 10])
        direction = np.random.choice([-1, 1])

        return torch.roll(sample, shifts=shift * direction, dims=axis)


class Identity:
    def __call__(self, sample):

        return sample


@torch.no_grad()
def init_weights(m, initialization='normal', **kwargs):
    if initialization == 'normal':
        if type(m) == nn.Linear or type(m) == nn.Conv3d:
            m.weight = nn.init.kaiming_normal_(m.weight, **kwargs)

    elif initialization == 'uniform':
        if type(m) == nn.Linear or type(m) == nn.Conv3d:
            m.weight = nn.init.kaiming_uniform_(m.weight, **kwargs)


def names_from_json(*args):
    names = []
    for batch_dir in args:
        with open(batch_dir, 'r') as fhand:
            info = json.load(fhand)
            names.append(info['names'])

    return np.concatenate(names)


def load_data(dir_batch, path_to_csv, target_name, index_name_csv, size=None):
    with open(f'{dir_batch}/clean_names.json', 'r') as fhand:
        names = json.load(fhand)['names']

    df = pd.read_csv(path_to_csv)
    df.set_index(index_name_csv, inplace=True)

    y = df.loc[names, target_name].values.astype('float32')
    X = np.load(f'{dir_batch}/clean_voxels.npy', mmap_mode='c')

    return X[:size], y[:size]
