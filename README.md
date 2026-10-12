# Funghi: MobileNetV3-Small

Training su PC con GPU o Kaggle; Raspberry Pi destinata all'inferenza. Python 3.11/3.12 consigliato. Questi script non contengono pesi già addestrati sui funghi.

## Installazione

Creare un ambiente virtuale. Installare una coppia compatibile **torch/torchvision** tramite https://pytorch.org/get-started/locally/ scegliendo la propria GPU. Per CPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Su Kaggle torch/torchvision sono normalmente preinstallati: non sostituirli senza necessità. Internet serve per i pesi ImageNet e il download del dataset.

## Dataset

```bash
python download_dataset.py --output data/mushrooms
```

Scarica la versione 3 di `tinkhoav/mushroom-classification`. Kaggle dichiara circa 2,70 GB: prevedere spazio aggiuntivo per download, estrazione e checkpoint. Se Kaggle richiede login, usare il suo flusso ufficiale; non inserire token nel codice.

La cartella passata a `--data` deve contenere direttamente una sottocartella per ogni classe: `images/NOME_CLASSE/foto.jpg`. Su Kaggle collegare il dataset al notebook e usare `/kaggle/input/mushroom-classification/images`.

## Allenamento

```bash
python train.py --data data/mushrooms/images --output runs/mobilenet --epochs 40 --batch-size 32
```

Su Kaggle:

```bash
python train.py --data /kaggle/input/mushroom-classification/images --output /kaggle/working/mobilenet --epochs 40 --batch-size 32
```

Se manca memoria GPU ridurre batch-size a 16/8; su Windows, se i worker danno problemi, aggiungere `--workers 0`. Per ripetere l'esperimento usare una nuova directory output.

Si parte realmente dai pesi ImageNet `IMAGENET1K_V1` e si aggiorna tutta la rete con AdamW, LR 3e-4, cosine decay e early stopping sulla loss di validazione. `--no-pretrained` è solo per test o training da zero. AMP è attivo soltanto su CUDA. Seed 36; risultati GPU non garantiti identici bit per bit.

Split stratificato circa 70/15/15 per classe, con almeno un esempio in validation e test. Classi con meno di 3 immagini causano un errore esplicito. Percorsi, hash e assegnazioni sono salvati in `split.json`. Si interrompe se uno stesso file immagine (hash identico) compare in split differenti: rimuovere i duplicati prima di ricominciare. Questo controllo non rileva ritagli, foto quasi identiche o immagini della stessa osservazione: per una valutazione scientifica raggruppare per osservazione/soggetto se disponibili i metadata.

Output: `best.pt`, `last.pt`, `labels.json`, `preprocess.json`, `split.json`, `history.json`, `config.json`. I checkpoint contengono pesi, classi ordinate e preprocessing. `last.pt` salva pesi dell'ultima epoca, **non** lo stato per ripresa esatta di optimizer/scheduler. Il numero di classi viene letto dalle cartelle, non imposto a priori.

## Test indipendente

```bash
python evaluate.py --data data/mushrooms/images --checkpoint runs/mobilenet/best.pt --split runs/mobilenet/split.json --output runs/mobilenet/test_metrics.json
```

Ricarica il modello migliore e verifica gli hash del test set. Salva accuracy, macro F1, balanced accuracy, metriche per classe e matrice di confusione. Non usare ripetutamente il test per scegliere gli iperparametri.

## Esportazione e Raspberry

Sul PC:

```bash
pip install -r requirements-export.txt
python export_onnx.py --checkpoint runs/mobilenet/best.pt --output deploy
python predict_onnx.py foto.jpg --model-dir deploy
```

L'export FP32 usa batch fisso 1, RGB, resize lato corto 256, center crop 224, normalizzazione ImageNet e confronta numericamente i logits ONNX/PyTorch. Non effettua quantizzazione INT8.

Copiare `deploy/` e `predict_onnx.py` sulla Pi. Serve un runtime ONNX compatibile con sistema operativo, Python e CPU Cortex-A53; verificare le wheel disponibili per il proprio Raspberry Pi OS (preferibilmente 64 bit). Installare `numpy Pillow onnxruntime` se esistono wheel compatibili, quindi eseguire lo stesso comando. Non è stata misurata la latenza sulla Pi 3. La probabilità softmax non è una misura calibrata dell'affidabilità e il modello sceglie sempre tra le classi note.

Fonti: https://www.kaggle.com/datasets/tinkhoav/mushroom-classification ; https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
