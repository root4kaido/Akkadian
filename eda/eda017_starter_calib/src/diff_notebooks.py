import json

with open('docs/survey/notebooks/downloaded/takamichitoda__dpc-starter-train/dpc-starter-train.ipynb') as f:
    latest = json.load(f)
with open('docs/survey/notebooks/downloaded/takamichitoda__dpc-starter-train/dpc-starter-train-v6.ipynb') as f:
    v6 = json.load(f)

print(f'v6 cells: {len(v6["cells"])}')
print(f'latest cells: {len(latest["cells"])}')

v6_code = []
for c in v6['cells']:
    if c['cell_type'] == 'code':
        v6_code.append(''.join(c['source']))

latest_code = []
for c in latest['cells']:
    if c['cell_type'] == 'code':
        latest_code.append(''.join(c['source']))

print(f'v6 code cells: {len(v6_code)}')
print(f'latest code cells: {len(latest_code)}')

keywords = ['LEARNING_RATE', 'EPOCHS', 'MAX_LENGTH', 'batch_size', 'gradient_accum',
            'label_smooth', 'generation_max', 'weight_decay', 'num_beams', 'load_best']

print()
print('=== v6 key params ===')
for code in v6_code:
    for line in code.split('\n'):
        if any(k in line for k in keywords):
            print(line.strip())

print()
print('=== latest key params ===')
for code in latest_code:
    for line in code.split('\n'):
        if any(k in line for k in keywords):
            print(line.strip())
