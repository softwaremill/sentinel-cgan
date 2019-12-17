from enum import Enum
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import reshape_as_image


class Purpose(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'


class SentinelDataGenerator():
    dataset: str
    landcover_mask_shape: Tuple[int, int]
    satellite_image_shape: Tuple[int, int]
    descriptor: str

    # TODO: validate 'data' directory structure
    def __init__(self,
                 dataset: str,
                 descriptor: str = 'data_descriptor.csv',
                 landcover_mask_shape=(1, 128, 128),
                 satellite_image_shape=(4, 128, 128)):
        self.dataset = dataset
        self.descriptor = descriptor
        self.landcover_mask_shape = landcover_mask_shape
        self.satellite_image_shape = satellite_image_shape

    def images_df(self, purpose: Purpose = Purpose.TRAIN):
        return pd.read_csv('../data/%s/%s/%s' % (self.dataset, purpose.value, self.descriptor))

    # TODO: augment data
    def load(self, batch: int = 1, purpose: Purpose = Purpose.TRAIN,
             landcover_mask_resampling: Optional[Resampling] = None,
             satellite_image_resampling: Optional[Resampling] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        images_df = self.images_df(purpose)
        images_df = images_df.sample(frac=1)

        for batch_df in [images_df[i:i + batch] for i in range(0, images_df.shape[0], batch)]:

            satellite_images = []
            landcover_masks = []

            for _, row in batch_df.iterrows():
                row_id = row['id']
                satellite_image_path = '../data/%s/%s/S/S_%s.tif' % (self.dataset, purpose.value, row_id)
                landcover_mask_path = '../data/%s/%s/LC/LC_%s.tif' % (self.dataset, purpose.value, row_id)

                satellite_image = self.read_raster(satellite_image_path, self.satellite_image_shape,
                                                   satellite_image_resampling)
                landcover_mask = self.read_raster(landcover_mask_path, self.landcover_mask_shape,
                                                  landcover_mask_resampling)

                satellite_images.append(satellite_image)
                landcover_masks.append(landcover_mask)

            yield np.array(satellite_images), np.array(landcover_masks)

    @staticmethod
    def read_raster(path: str, out_shape: Tuple[int, int], resampling: Optional[Resampling] = None, clip: Optional[int] = None) -> np.ndarray:
        if resampling:
            raster = rasterio.open(path, dtype='int16').read(out_shape=out_shape, resampling=resampling)
        else:
            raster = rasterio.open(path, dtype='int16').read(out_shape=out_shape)

        raster = np.nan_to_num(raster, posinf=0, neginf=0)
        raster = np.clip(raster, 0, clip) if clip else raster
        return reshape_as_image(raster)
