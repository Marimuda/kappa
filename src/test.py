import os 
import sys
import wandb
import argparse
import json 

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

args = parser.parse_args()

"""Dataset"""
base_path = "/dtu/blackhole/14/189044/atde/challenge_data/test"
save_path = "/dtu/blackhole/14/189044/atde/challenge_model/"
RUN_NAME = f'resnet_lr_{args.lr}_b_{args.batch_size}'

model_path = os.path.join(save_path, f'{RUN_NAME}.pth')

crop_factory = VertebraDatasetFactory(base_path=base_path, return_sample_id=True)
crop_dataset = crop_factory.create_dataset(dataset_type='crop')
crop_loader = DataLoader(crop_dataset, batch_size=16, shuffle=False)

"""Model"""
resnet = build_resnet(model_depth=18, n_input_channels=1, n_classes=1)
resnet.to(device)
resnet.load_state_dict(torch.load(model_path), strict=True)
"""Inferece"""

results = []

for img_volume, _, sample_ids in tqdm(crop_loader, desc=f'Inferece'):
    # img_volume: torch.Size([b, 241, 241, 241])
    img_volume = img_volume.unsqueeze(1) # MAKE CHANNLES 1
    img_volume = img_volume.to(device)
    with torch.no_grad():
        pred_logits = resnet(img_volume).squeeze()
    pred_conf = nn.Sigmoid()(pred_logits)
    pred_class = pred_conf.round().tolist()

    for sid, pclass, pconf in zip(sample_ids, pred_class, pred_conf):
    
        curr_data = {
            "scan_id": sid,
            "outlier": int(pclass),
            "outlier_probability": pconf.item(),
            "outlier_threshold": 0.5
        }
        results.append(curr_data)
        
    

with open("baseline.json", "w") as f:
    json.dump(results, f)


# import numpy as np
# all_predictions = np.array(all_predictions)
# all_labels = np.array(all_labels)

# """
# Zeros: 1092
# Ones: 3276
# """
# breakpoint()
# acc = sum(np.array(all_predictions == all_labels))/ len(all_labels)
# print(f'Acc = {acc:.2f}')