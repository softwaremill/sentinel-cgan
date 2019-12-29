import os
import pandas as pd

grid = 'grid.shp'
sentinel = 'sentinel.tif'
land_cover = 'land_cover.tif'

os.makedirs('dataset/S', exist_ok=True)
os.makedirs('dataset/LC', exist_ok=True)

effective_ids = []

for i in range(1, 1000):
    code_s = os.system(
        'gdalwarp -cutline %s -cwhere "id=%s" -tr 10 10 -crop_to_cutline %s dataset/S/S_%s.tif' % (
            grid, i, sentinel, i))
    code_ls = os.system(
        'gdalwarp -cutline %s -cwhere "id=%s" -tr 100 100 -crop_to_cutline %s dataset/LC/LC_%s.tif' % (
            grid, i, land_cover, i))

    if code_s == 0 or code_ls == 0:
        effective_ids.append(i)

pd.DataFrame({
    'id': list(range(1, 1000))
}).to_csv('/tmp/data_descriptor.csv', index=False)
