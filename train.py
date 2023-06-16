import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms, datasets

import numpy as np
import wandb
import torchmetrics

from argparse import ArgumentParser
from torcheval.metrics import Throughput

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class MLP(nn.Module):

    def __init__(self, n_hidden, n_layers):
        super(MLP, self).__init__()

        self.mlp = nn.ModuleList()
        flattened_dim = 4*1*1
        self.mlp.append(nn.Sequential(nn.Conv2d(3, n_hidden, kernel_size=3, stride=2, padding=1), nn.Dropout(), nn.ReLU()))
        for n in range(n_layers):
            self.mlp.append(nn.Sequential(nn.Conv2d(n_hidden, n_hidden // 2, kernel_size=3, stride=2, padding=1), nn.Dropout(), nn.ReLU()))
            n_hidden = n_hidden // 2

        self.output = nn.Sequential(nn.Flatten(), nn.Linear(flattened_dim, 1))

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        
        print(x.shape)
        #raise ValueError()
        x = self.output(x)
        return x

def get_dataset(args):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train_dataset = datasets.PCAM(root="./", split='train', transform=transform, download=True)
    val_dataset = datasets.PCAM(root="./", split='val', transform=transform, download=True)

    train_dataloader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)
    
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)


    return train_dataloader, val_dataloader

@torch.no_grad()
def validate(args, model, criterion, dataloader):
    model.eval()
    F1 = torchmetrics.F1Score(task="binary")
    losses = []
    predictions = []
    Y = []
    for i, (x, y) in enumerate(dataloader):
        x , y = x.to(device), y.to(device).float()
        Y += y.cpu().tolist()
        output = model(x).squeeze(1)
        prediction = torch.sigmoid(output) > 0.5
        predictions += prediction.cpu().tolist()
        loss = criterion(output, y)
        losses.append(loss.item())

    f1 = F1(torch.tensor(predictions), torch.tensor(Y))
    wandb.log({'log-step': i, 'val-loss': np.array(losses).mean(), "f1-score": np.array(f1).mean()})

def train(args, model, train_dataloader, optimizer, criterion):
    model.train()
    losses = []
    metric = Throughput()
    ts = time.monotonic()
    for i, (x, y) in enumerate(train_dataloader):
        x , y = x.to(device), y.to(device).float()

        output = model(x).squeeze(1)

        loss = criterion(output, y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        if i%100==0:
            elapsed_time_sec = time.monotonic() - ts
            metric.update((i+1)*args.batch_size, elapsed_time_sec)
            throughput = metric.compute()
            wandb.log({'log-step': i, 'train-loss': np.array(losses[-100:]).mean(), "throughput imgs/sec":throughput})


def main(args):
    model = MLP(args.n_hidden, args.n_layers)
    model = torch.compile(model)
    model = model.to(device)

    train_dataloader, val_dataloader = get_dataset(args)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    wandb.init(project="PCAM-small-CNN")

    for i in range(10):
        train(args, model, train_dataloader, optimizer, criterion)
        validate(args, model, criterion, val_dataloader)

if __name__=="__main__":
    
    # Create the parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument('--data_path', type=str, help='Input file path')
    parser.add_argument('--n_layers', type=int, default=6, help='Enable verbose mode')
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--num_workers", type=int, default=16)


    # Parse the arguments
    args = parser.parse_args()

    main(args)