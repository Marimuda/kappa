import os 
import sys
import wandb
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# custom imports
from dataset.vertebra_db_factory_label import VertebraDBFactoryLabel
from models.resnet import generate_model as build_resnet


"""Argument loading"""
device = torch.device('cuda')

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50) 
parser.add_argument('--model-depth', type=int, default=18)

parser.add_argument('--debug', action='store_true', dest='debug')
parser.add_argument('--no-debug', action='store_false', dest='debug')
parser.set_defaults(debug=True)


parser.add_argument('--class-weights', action='store_true', dest='class_weights')
parser.add_argument('--no-class-weights', action='store_false', dest='class_weights')
parser.set_defaults(class_weights=False)

args = parser.parse_args()

"""Dataset"""
base_path = "/dtu/blackhole/14/189044/atde/challenge_data/train"
save_path = "/dtu/blackhole/14/189044/atde/challenge_model/"
RUN_NAME = f'resnet_segmentation_{args.model_depth}_lr_{args.lr}_b_{args.batch_size}'
if args.class_weights: 
    RUN_NAME += '_class_weights'
assert args.model_depth in [10, 18, 34, 50, 101, 152, 200]
if not args.debug:
    os.makedirs(save_path, exist_ok=True)
    logger = wandb.init(
        project=f"sm-2024", 
        entity="atde_",
        name=RUN_NAME,
    )
model_path = os.path.join(save_path, f'{RUN_NAME}.pth')

crop_factory = VertebraDBFactoryLabel(base_path=base_path)
crop_dataset = crop_factory.create_dataset(dataset_type='crop')
crop_loader = DataLoader(crop_dataset, batch_size=args.batch_size, shuffle=True)

"""Model"""
resnet = build_resnet(model_depth=args.model_depth, n_input_channels=2, n_classes=1)
resnet.to(device)
total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)

if args.class_weights:
    #class_weights = torch.tensor([, 0.25], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.33], device=device))
else :
    loss_fn = nn.BCEWithLogitsLoss()
sigmoid = nn.Sigmoid()
opt = torch.optim.AdamW(resnet.parameters(), lr=args.lr)
print(f'[INFO] total_params = {total_params/1e6:.2f}M')
print(f'[INFO] model_path = {model_path}')
for e in range(50):
    resnet.train()
    train_loss = 0
    train_acc = 0
    
    for img_volume, segmentation, label in tqdm(crop_loader, desc=f'[EPOCH {e+1}/50]'):
        # img_volume: torch.Size([b, 241, 241, 241])
        img_volume = img_volume.unsqueeze(1) # MAKE CHANNLES 1
        img_volume = img_volume.to(device)

        segmentation = segmentation.unsqueeze(1) # MAKE CHANNLES 1
        segmentation = segmentation.to(device)
        
        input = torch.cat([img_volume, segmentation], dim=1)
        label = label.to(device).unsqueeze(1)

        opt.zero_grad()
        pred_logits = resnet(input)
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

    