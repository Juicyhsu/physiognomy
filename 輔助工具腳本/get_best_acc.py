import json
from pathlib import Path

parts = ['eye', 'nose', 'brow', 'mouth', 'face']
exps = {
    'Exp A (MobileNetV2)': 'models_cnn_expA_mobilenet',
    'Exp B (EfficientNetB0)': 'models_cnn_expB_efficientnet',
    'Exp C (MobileNetV2 LowLR)': 'models_cnn_expC_mobilenet_lowLR',
    'Exp D (MobileNetV2 Batch16)': 'models_cnn_expD_mobilenet_batch16'
}

print('Part\tExp A\tExp B\tExp C\tExp D')
for p in parts:
    row = [p]
    for label, folder in exps.items():
        path = Path(folder) / f'{p}_history.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                val_acc = data.get('val_accuracy', [])
                best = max(val_acc) if val_acc else 0.0
                row.append(f'{best*100:.2f}%')
        else:
            row.append('N/A')
    print('\t'.join(row))
