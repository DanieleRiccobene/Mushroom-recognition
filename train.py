"""Fine-tuning completo; split stratificati salvati, selezione solo su validation."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from common import model_for, transform, PREPROCESS, write_json

def split_dataset(ds,seed):
    rng=random.Random(seed); splits={'train':[],'val':[],'test':[]}
    for cid,name in enumerate(ds.classes):
        ids=[i for i,y in enumerate(ds.targets) if y==cid]
        if len(ids)<3: raise ValueError(f'{name}: almeno 3 immagini richieste, trovate {len(ids)}')
        rng.shuffle(ids); nv=max(1,round(len(ids)*.15)); nt=max(1,round(len(ids)*.15))
        splits['val']+=ids[:nv];splits['test']+=ids[nv:nv+nt];splits['train']+=ids[nv+nt:]
    return splits

def run_epoch(model,loader,device,optimizer=None,scaler=None):
    training=optimizer is not None;model.train(training)
    total=correct=0;loss_sum=0.
    with torch.set_grad_enabled(training):
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(x);loss=nn.functional.cross_entropy(logits,y)
            if training:
                scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total+=y.numel();correct+=(logits.argmax(1)==y).sum().item();loss_sum+=loss.item()*y.numel()
    return {'loss':loss_sum/total,'accuracy':correct/total}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--output',default='runs/mobilenet')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=36)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--patience',type=int,default=8)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--no-pretrained',action='store_true',help='Solo per test o training da zero')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<2 or a.patience<1: p.error('epochs e patience >=1; batch-size >=2')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise ValueError('Output non vuoto: scegli una nuova directory per non sovrascrivere un esperimento.')
    out.mkdir(parents=True,exist_ok=True)
    root=Path(a.data).resolve();base=ImageFolder(root)
    if len(base.classes)<2:raise ValueError('Servono almeno due classi nella cartella --data')
    splits=split_dataset(base,a.seed)
    # Hashes permit evaluation to reject a changed dataset and expose exact duplicates across splits.
    records=[];seen={}
    for split,ids in splits.items():
        for i in ids:
            path,label=base.samples[i]; digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest in seen and seen[digest]!=split:
                raise ValueError('Duplicato identico tra split: deduplica il dataset prima del training: '+path)
            seen[digest]=split
            records.append({'path':Path(path).relative_to(root).as_posix(),'label':label,'split':split,'sha256':digest})
    write_json(out/'split.json',{'classes':base.classes,'seed':a.seed,'records':records})
    write_json(out/'labels.json',{str(i):c for i,c in enumerate(base.classes)})
    write_json(out/'config.json',vars(a));write_json(out/'preprocess.json',PREPROCESS)
    loaders={}
    for name in ['train','val']:
        ds=Subset(ImageFolder(root,transform=transform(name=='train')),splits[name])
        # BatchNorm classifier cannot train on a singleton final batch.
        drop=name=='train' and len(ds)%a.batch_size==1
        if name=='train' and len(ds)<2:raise ValueError('Training set troppo piccolo')
        loaders[name]=DataLoader(ds,batch_size=a.batch_size,shuffle=name=='train',num_workers=a.workers,drop_last=drop,pin_memory=device.type=='cuda')
    model=model_for(len(base.classes),not a.no_pretrained).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=float('inf');stale=0;history=[]
    print(f'Device={device}; classi={len(base.classes)}; split='+str({k:len(v) for k,v in splits.items()}),flush=True)
    for epoch in range(a.epochs):
        tr=run_epoch(model,loaders['train'],device,optimizer,scaler)
        va=run_epoch(model,loaders['val'],device)
        scheduler.step();improved=va['loss']<best
        if improved:best=va['loss'];stale=0
        else:stale+=1
        ck={'architecture':'mobilenet_v3_small','classes':base.classes,'class_to_idx':base.class_to_idx,'preprocess':PREPROCESS,'epoch':epoch+1,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'validation':va,'torch_version':str(torch.__version__)}
        if improved:torch.save(ck,out/'best.pt')
        torch.save(ck,out/'last.pt')
        history.append({'epoch':epoch+1,'train':tr,'val':va});write_json(out/'history.json',history)
        print(f'Epoca {epoch+1}: train={tr}, val={va}',flush=True)
        if stale>=a.patience:break
    print('Training terminato. Valuta best.pt con evaluate.py: il test non è stato usato per selezionare il modello.')

if __name__=='__main__':main()
