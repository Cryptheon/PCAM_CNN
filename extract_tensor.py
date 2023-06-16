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
        flattened_dim = 16*3*3
        self.mlp.append(nn.Sequential(nn.Conv2d(3, n_hidden, kernel_size=3, stride=2, padding=1), nn.Dropout(), nn.ReLU()))
        for n in range(n_layers):
            self.mlp.append(nn.Sequential(nn.Conv2d(n_hidden, n_hidden // 2, kernel_size=3, stride=2, padding=1), nn.Dropout(), nn.ReLU()))
            n_hidden = n_hidden // 2

        self.flatten = nn.Flatten()
        self.map = nn.Linear(flattened_dim, 48)
        self.relu = nn.ReLU()
        self.out = nn.Linear(48,1)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        
        # get only the 48 sized latents
        x = self.map(self.flatten(x))
        return x

def get_dataset(args):
    
    val_transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train_dataset = datasets.PCAM(root="./", split='train', transform=val_transform, download=True)
    val_dataset = datasets.PCAM(root="./", split='val', transform=val_transform, download=True)

    train_dataloader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)
    
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.num_workers)


    return train_dataloader, val_dataloader

@torch.no_grad()
def validate(args, model, dataloader):
    model.eval()
    latents = []
    Y = []
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        Y += y.cpu().tolist()
        output = model(x).squeeze(1).cpu().numpy()
        latents.append(output)
    return latents, Y

@torch.no_grad()
def train(args, model, train_dataloader):
    latents = []
    Y = []
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)

        Y += y.cpu().tolist()
        output = model(x).squeeze(1).cpu().numpy()
        latents.append(output)

    return latents, Y

def main(args):
    model = MLP(args.n_hidden, args.n_layers)
    model = model.to(device)
    state_dict = torch.load("./models/48_latent_model_0.85.pt")
    
    state_dict = {k.replace("_orig_mod.", ""): v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    train_dataloader, val_dataloader = get_dataset(args)

    train_latents, y_train = train(args, model, train_dataloader)
    val_latents, y_val = validate(args, model, val_dataloader)

    train_latents = np.vstack(train_latents)
    val_latents = np.vstack(val_latents)

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    np.save('./latents/train_latents_48.npy', train_latents)
    np.save('./latents/val_latents_48.npy', val_latents)
    np.save('./latents/train_y.npy', y_train)
    np.save('./latents/val_y.npy', y_val)


if __name__=="__main__":
    
    # Create the parser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument('--data_path', type=str, help='Input file path')
    parser.add_argument('--n_layers', type=int, default=4, help='Enable verbose mode')
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)


    parser.add_argument("--num_workers", type=int, default=16)


    # Parse the arguments
    args = parser.parse_args()

    main(args)