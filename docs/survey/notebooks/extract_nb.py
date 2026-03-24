import json

with open('/home/user/work/Akkadian/docs/survey/notebooks/lb-35-9-ensembling-post-processing-baseline.ipynb') as f:
    nb = json.load(f)

with open('/home/user/work/Akkadian/docs/survey/notebooks/lb-35-9-nb-extracted.txt', 'w') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"=== Cell {i} ({cell['cell_type']}) ===\n")
        out.write(''.join(cell['source']))
        out.write('\n\n')
