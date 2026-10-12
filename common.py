import json
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms

PREPROCESS = {'resize_shorter':256,'crop':224,'interpolation':'bilinear','color':'RGB','scale':255.0,'mean':[0.485,0.456,0.406],'std':[0.229,0.224,0.225]}

def model_for(n, pretrained=False):
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model=models.mobilenet_v3_small(weights=weights)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,n)
    return model

def transform(train=False):
    ops=([transforms.RandomResizedCrop(224,scale=(0.65,1.0)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.15,.15,.15,.03)] if train else [transforms.Resize(256),transforms.CenterCrop(224)])
    return transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize(PREPROCESS['mean'],PREPROCESS['std'])])

def load_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=model_for(len(ck['classes']))
    model.load_state_dict(ck['model'])
    return model.eval(),ck

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
