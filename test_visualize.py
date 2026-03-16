"""Run segmentation + save CSV + visualize for core 005."""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import csv
from kandus_method.tissue_segmentation import segment_tissue

core_id = '005_r1c5'
he_path = f'data_raw/02-008_HE_A12_v2_s13/02-008_HE_A12_v2_s13_{core_id}.jpg.jpeg'

print(f'Segmenting {core_id}...')
result = segment_tissue(he_path)

tc = result.get('tc_coords', [])
lc = result.get('lc_coords', [])
st = result.get('st_coords', [])
print(f'TC={len(tc)} LC={len(lc)} ST={len(st)}')

# Save CSV
Path('results/coords').mkdir(parents=True, exist_ok=True)
csv_path = Path(f'results/coords/{core_id}_cells.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['x','y','area_px','cell_type'])
    w.writeheader()
    for c in tc: w.writerow({'x':c['x'],'y':c['y'],'area_px':c.get('area_px',''),'cell_type':'TC'})
    for c in lc: w.writerow({'x':c['x'],'y':c['y'],'area_px':c.get('area_px',''),'cell_type':'LC'})
    for c in st: w.writerow({'x':c['x'],'y':c['y'],'area_px':'','cell_type':'ST'})
print(f'Saved CSV: {csv_path}')

# Now visualize
import visualize_coords
visualize_coords.visualize_core(core_id)
