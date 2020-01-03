import tarfile
import tempfile
from pathlib import Path

import boto3
import tqdm
from botocore.handlers import disable_signing

bucket_name = 'sml-ml-data-sets'
key = 'sentinel-cgan/bdot.tar.gz'
data_dir = Path('../data')
s3 = boto3.resource('s3', region_name='eu-central-1')
s3.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)


def hook(t):
    def inner(bytes_amount):
        t.update(bytes_amount)

    return inner


with tempfile.TemporaryDirectory() as temp_dir:
    temp_archive = str(Path(temp_dir).joinpath('bdot.tar.gz').resolve())

    file_object = s3.Object(bucket_name, key)
    file_size = file_object.content_length

    with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=temp_archive) as t:
        s3.Bucket(bucket_name).download_file(key, temp_archive, Callback=hook(t))

    with tarfile.open(temp_archive) as tar:
        tar.extractall(path=data_dir)
