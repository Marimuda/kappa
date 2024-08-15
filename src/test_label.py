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
from dataset.vertebra_db_factory_label import VertebraDBFactoryLabel
from models.resnet import generate_model as build_resnet


"""Argument loading"""
device = torch.device('cuda')

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--model-depth', type=int, default=18)

parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--class-weights', action='store_true', dest='class_weights')
parser.add_argument('--no-class-weights', action='store_false', dest='class_weights')
parser.set_defaults(class_weights=False)

args = parser.parse_args()

"""Dataset"""
base_path = "/dtu/blackhole/14/189044/atde/challenge_data/test"
save_path = "/dtu/blackhole/14/189044/atde/challenge_model/"
RUN_NAME = f'resnet_segmentation_{args.model_depth}_lr_{args.lr}_b_{args.batch_size}'

if args.class_weights: 
    RUN_NAME += '_class_weights'

model_path = os.path.join(save_path, f'{RUN_NAME}.pth')

crop_factory = VertebraDBFactoryLabel(base_path=base_path, return_sample_id=True)
crop_dataset = crop_factory.create_dataset(dataset_type='crop')
crop_loader = DataLoader(crop_dataset, batch_size=16, shuffle=False)

"""Model"""
resnet = build_resnet(model_depth=18, n_input_channels=2, n_classes=1)
resnet.to(device)
resnet.load_state_dict(torch.load(model_path), strict=True)
"""Inferece"""

results = []

for img_volume, segmentation, _, sample_ids in tqdm(crop_loader, desc=f'Inferece'):
    # img_volume: torch.Size([b, 241, 241, 241])
    img_volume = img_volume.unsqueeze(1)
    img_volume = img_volume.to(device)

    segmentation = segmentation.unsqueeze(1) # MAKE CHANNLES 1
    segmentation = segmentation.to(device)
    
    input = torch.cat([img_volume, segmentation], dim=1)
    with torch.no_grad():
        pred_logits = resnet(input).squeeze()
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
        #breakpoint()
        
    
res_name = 'bone_voyage_v4'
if args.class_weights: 
    res_name += '_class_weights'

with open(f"{res_name}.json", "w") as f:
    json.dump(results, f)

