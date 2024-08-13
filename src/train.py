import os 
import sys
# import wandb
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# custom imports
from dataset.vertebra_dataset_factory import VertebraDatasetFactory
from models.resnet import generate_model as build_resnet


"""Argument loading"""
device = torch.device('cuda')

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--debug', action='store_true', dest='debug')
parser.add_argument('--no-debug', action='store_false', dest='debug')
parser.set_defaults(debug=True)

args = parser.parse_args()

"""Dataset"""
base_path = "/dtu/blackhole/14/189044/atde/challenge_data/train"
save_path = "/dtu/blackhole/14/189044/atde/challenge_model/"
RUN_NAME = f'resnet_lr_{args.lr}_b_{args.batch_size}'

if not args.debug:
    os.makedirs(save_path, exist_ok=True)
    logger = wandb.init(
        project=f"sm-2024", 
        entity="atde_",
        name=RUN_NAME,
    )
model_path = os.path.join(save_path, f'{RUN_NAME}.pth')

crop_factory = VertebraDatasetFactory(base_path=base_path)
crop_dataset = crop_factory.create_dataset(dataset_type='crop')
crop_loader = DataLoader(crop_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

"""Model"""
resnet = build_resnet(model_depth=18, n_input_channels=1, n_classes=1)
resnet.to(device)
total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)

loss_fn = nn.BCEWithLogitsLoss()
sigmoid = nn.Sigmoid()
opt = torch.optim.AdamW(resnet.parameters(), lr=args.lr)
print(f'[INFO] total_params = {total_params/1e6:.2f}M')

for e in range(50):
    resnet.train()
    train_loss = 0
    train_acc = 0
    
    for img_volume, label in tqdm(crop_loader, desc=f'[EPOCH {e+1}/50]'):
        # img_volume: torch.Size([b, 241, 241, 241])
        img_volume = img_volume.unsqueeze(1) # MAKE CHANNLES 1
        img_volume = img_volume.to(device)
        label = label.to(device)

        opt.zero_grad()
        pred_logits = resnet(img_volume).squeeze()
        batch_loss = loss_fn(pred_logits, label)
        batch_loss.backward()
        opt.step()

        pred_class = sigmoid(pred_logits).round()

        train_loss += batch_loss
        train_acc += sum(pred_class == label)/label.shape[0]
        
    train_loss /= len(crop_loader)
    train_acc /= len(crop_loader)

    if not args.debug:
        torch.save(resnet.state_dict(), model_path)
        logger.log({
            'Train Loss': train_loss,
            'Train acc': train_acc
        })
    else :
        print(f'Loss : {train_loss:2f}')
        print(f'Acc: {train_acc:.2f}')

    