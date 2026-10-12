"""Inferenza CPU senza PyTorch; Pillow + NumPy + ONNX Runtime."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import onnxruntime as ort
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--model-dir',default='deploy');p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    root=Path(a.model_dir);meta=json.loads((root/'metadata.json').read_text());cfg=meta['preprocess']
    with Image.open(a.image) as src:
        im=src.convert('RGB');w,h=im.size;s=cfg['resize_shorter']
        # Identico resize e arrotondamento del CenterCrop torchvision.
        size=(s,int(s*h/w)) if w<=h else (int(s*w/h),s)
        im=im.resize(size,Image.Resampling.BILINEAR);crop=cfg['crop'];left=round((im.width-crop)/2);top=round((im.height-crop)/2)
        im=im.crop((left,top,left+crop,top+crop));x=np.asarray(im,dtype=np.float32)/cfg['scale']
    x=(x-np.array(cfg['mean'],dtype=np.float32))/np.array(cfg['std'],dtype=np.float32)
    x=np.ascontiguousarray(x.transpose(2,0,1)[None]);options=ort.SessionOptions();options.intra_op_num_threads=a.threads
    sess=ort.InferenceSession(str(root/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    start=time.perf_counter();z=sess.run(None,{'images':x})[0][0];elapsed=time.perf_counter()-start
    prob=np.exp(z-z.max());prob/=prob.sum()
    for i in np.argsort(prob)[::-1][:5]:print(f'{i}: {meta["classes"][i]} — {prob[i]:.2%}')
    print(f'Inferenza (prima chiamata, esclusi preprocessing e caricamento): {elapsed:.3f} s')
