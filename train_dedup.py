"""Avvio compatibile con il pacchetto mushroom_mobilenet originale.
Uso: python train_dedup.py --data data/mushrooms/images --output runs/mobilenet_dedup --epochs 40 --batch-size 32
Non modifica train.py e non cancella immagini. Deduplica SHA-256 prima dello split.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def deduplicated_split(ds, seed, report_path):
    root = Path(ds.root).resolve()
    groups = {}
    for i, (path, label) in enumerate(ds.samples):
        with open(path, 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        groups.setdefault(digest, []).append(i)
        if (i + 1) % 5000 == 0:
            print(f'Controllo duplicati: {i + 1}/{len(ds.samples)}', flush=True)

    kept = {cid: [] for cid in range(len(ds.classes))}
    duplicates, conflicts = [], []
    def record(i):
        path, label = ds.samples[i]
        return {'path': Path(path).resolve().relative_to(root).as_posix(),
                'label': label, 'class': ds.classes[label]}
    for digest, ids in groups.items():
        labels = {ds.samples[i][1] for i in ids}
        if len(labels) > 1:
            conflicts.append({'sha256': digest, 'excluded': [record(i) for i in ids]})
            continue
        # Stable representative, independent of discovery order.
        ids = sorted(ids, key=lambda i: str(ds.samples[i][0]))
        keep = ids[0]
        kept[ds.samples[keep][1]].append(keep)
        if len(ids) > 1:
            duplicates.append({'sha256': digest, 'kept': record(keep),
                               'excluded': [record(i) for i in ids[1:]]})

    counts = {name: len(kept[cid]) for cid, name in enumerate(ds.classes)}
    report = {'method': 'SHA-256 of file bytes before stratified splitting',
              'original_files': len(ds.samples), 'retained_files': sum(counts.values()),
              'same_label_copies_excluded': sum(len(g['excluded']) for g in duplicates),
              'conflicting_label_files_excluded': sum(len(g['excluded']) for g in conflicts),
              'retained_per_class': counts, 'duplicates': duplicates, 'conflicts': conflicts,
              'files_deleted': 0}
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"File originali: {report['original_files']}; unici utilizzabili: {report['retained_files']}; "
          f"copie escluse: {report['same_label_copies_excluded']}; "
          f"file con etichette in conflitto: {report['conflicting_label_files_excluded']}", flush=True)
    rare = {name: n for name, n in counts.items() if n < 3}
    if rare:
        raise ValueError('Meno di 3 immagini uniche utilizzabili in alcune classi. '
                         'Consulta dedup_report.json; nessuna classe è stata rimappata: ' + str(rare))
    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for cid in range(len(ds.classes)):
        ids = sorted(kept[cid]); rng.shuffle(ids)
        nv = max(1, round(len(ids) * .15)); nt = max(1, round(len(ids) * .15))
        splits['val'].extend(ids[:nv])
        splits['test'].extend(ids[nv:nv + nt])
        splits['train'].extend(ids[nv + nt:])
    return splits


def main():
    import train
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', default='runs/mobilenet')
    args, _ = parser.parse_known_args()
    def split(ds, seed):
        return deduplicated_split(ds, seed, Path(args.output) / 'dedup_report.json')
    train.split_dataset = split
    train.main()


if __name__ == '__main__':
    main()
