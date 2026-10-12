"""FP32, input fisso 1x3x224x224; verifica numerica PyTorch/ONNX Runtime."""
import argparse
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from common import load_checkpoint,write_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='deploy');a=p.parse_args()
    model,ck=load_checkpoint(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    x=torch.randn(1,3,224,224)
    torch.onnx.export(model,x,str(out/'model.onnx'),input_names=['images'],output_names=['logits'],opset_version=17,dynamo=False)
    onnx.checker.check_model(str(out/'model.onnx'))
    sess=ort.InferenceSession(str(out/'model.onnx'),providers=['CPUExecutionProvider'])
    with torch.inference_mode():reference=model(x).numpy()
    actual=sess.run(None,{'images':x.numpy()})[0]
    np.testing.assert_allclose(actual,reference,rtol=1e-3,atol=1e-5)
    write_json(out/'metadata.json',{'classes':ck['classes'],'preprocess':ck['preprocess'],'architecture':ck['architecture']})
    print('Export verificato:',out,'massimo errore assoluto:',float(np.max(np.abs(actual-reference))))
