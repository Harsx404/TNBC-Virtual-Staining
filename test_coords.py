"""Quick test: segment one core and check coordinate output."""
import sys; sys.path.insert(0, '.')
from kandus_method.tissue_segmentation import segment_tissue, extract_cell_coordinates

he_path = r'data_raw/02-008_HE_A12_v2_s13/02-008_HE_A12_v2_s13_005_r1c5.jpg.jpeg'

print('Segmenting core 005...')
result = segment_tissue(he_path)
coords = extract_cell_coordinates(result)

print(f"\nTC cells: {len(coords['TC'])}")
print(f"LC cells: {len(coords['LC'])}")
print(f"ST sample points: {len(coords['ST'])}")

print("\nFirst 3 TC coords:")
for c in coords['TC'][:3]:
    print(f"  x={c['x']}, y={c['y']}, area_px={c['area_px']}")

print("\nFirst 3 LC coords:")
for c in coords['LC'][:3]:
    print(f"  x={c['x']}, y={c['y']}, area_px={c['area_px']}")

print("\nFirst 3 ST sample points:")
for c in coords['ST'][:3]:
    print(f"  x={c['x']}, y={c['y']}")
