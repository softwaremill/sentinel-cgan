import os
import pandas as pd

dir = './'

grid = os.path.join(dir, 'grid_c.shp')
sentinel = os.path.join(dir, 'sentinel.tif')
land_cover = os.path.join(dir, 'land_cover.tif')

dataset = os.path.join(dir, 'dataset')
s_dataset = os.path.join(dataset, 'S')
lc_dataset = os.path.join(dataset, 'LC')

os.makedirs(s_dataset, exist_ok=True)
os.makedirs(lc_dataset, exist_ok=True)

effective_ids = []

for i in range(1, 1000):
    code_s = os.system(
        'gdalwarp -cutline %s -cwhere "id=%s" -tr 10 10 -crop_to_cutline %s %s/S_%s.tif' % (
            grid, i, sentinel, s_dataset, i))
    code_ls = os.system(
        'gdalwarp -cutline %s -cwhere "id=%s" -tr 100 100 -crop_to_cutline %s %s/LC_%s.tif' % (
            grid, i, land_cover, lc_dataset, i))

    if code_s == 0 or code_ls == 0:
        effective_ids.append(i)
    else:
        break

pd.DataFrame({
    'id': effective_ids
}).to_csv('%s/data_descriptor.csv' % dataset, index=False)
