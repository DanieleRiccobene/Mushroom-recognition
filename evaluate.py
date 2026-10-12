import argparse,hashlib,json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from common import load_checkpoint,transform,write_json
class HeldOut(Dataset):
    def __init__(self,root,records):self.root=Path(root);self.records=records;self.tf=transform()
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(self.root/r['path']) as im:x=self.tf(im.convert('RGB'))
        return x,r['label']
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--output',default='test_metrics.json');p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
    model,ck=load_checkpoint(a.checkpoint);s=json.loads(Path(a.split).read_text())
    if s['classes']!=ck['classes']:raise ValueError('Ordine delle classi non corrispondente')
    records=[r for r in s['records'] if r['split']=='test']
    for r in records:
        if hashlib.sha256((Path(a.data)/r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Dataset modificato: '+r['path'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    n=len(ck['classes']);cm=torch.zeros(n,n,dtype=torch.int64)
    with torch.inference_mode():
        for x,y in DataLoader(HeldOut(a.data,records),batch_size=a.batch_size):
            pred=model(x.to(device)).argmax(1).cpu();cm+=torch.bincount(y*n+pred,minlength=n*n).reshape(n,n)
    tp=cm.diag().float();support=cm.sum(1);precision=tp/cm.sum(0).clamp_min(1);recall=tp/support.clamp_min(1);f1=2*precision*recall/(precision+recall).clamp_min(1e-12)
    result={'samples':len(records),'accuracy':tp.sum().item()/len(records),'macro_f1':f1.mean().item(),'balanced_accuracy':recall.mean().item(),'per_class':{name:{'support':int(support[i]),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,name in enumerate(ck['classes'])},'confusion_matrix':cm.tolist()}
    write_json(a.output,result);print({k:result[k] for k in ['samples','accuracy','macro_f1','balanced_accuracy']})
if __name__=='__main__':main()
