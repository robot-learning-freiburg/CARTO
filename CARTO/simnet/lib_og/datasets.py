import io

# import json
# import logging
# import pathlib
# from concurrent import futures
import tarfile

# import uuid
# import os
# import pickle
# import random
# import collections

# import numpy as np
# import boto3
import zstandard as zstd

# import tqdm
# import IPython
# from typing import Any, Dict, List, Callable

# from CARTO.simnet.infra import shapenetcore

# S3_BUCKET = 'mmt-learning-data'

# DISABLE_LOCAL_TEST_FILES = False

# RUN_ID = uuid.uuid4().hex


# def get_local_cache_path(name):
#   return pathlib.Path(f'data/cache/{name}')


# class MMTAnnotations:
#   TEST_IDS = [
#       'Bottle',
#       'Refrigerator',
#   ]

#   def __init__(self):
#     self.dataset_path = get_datasets_path() / 'mmt_annotations'
#     self.json_path = self.dataset_path / 'index.json'
#     self.index = read_compressed_json(self.dataset_path / 'index.json.zst')
#     self.id_index = {meta['id']: meta for meta in self.index}

#   def get_sample_meta(self, id_):
#     if id_ not in self.id_index:
#       raise ValueError(f'id_={id_} not in index')
#     return self.id_index[id_]

#   def is_object_instance_only(self, class_id):
#     if class_id in ['Table', 'Lamp', 'Bag', 'Display', 'Can']:
#       return True
#     return False

#   def is_object_alignable(self, class_id):
#     if class_id in ['Camera']:
#       return True
#     return False

#   def is_class_annotated(self, class_id):
#     return class_id in self.id_index

#   def sample_enforced_corr_pair(self, class_id, anno_id='_0000', find_anno_id=False):
#     for objects in self.index:
#       if objects['class'] == class_id:
#         objects_per_class_id = objects
#     assert objects_per_class_id is not None
#     objects = objects_per_class_id['partnet_data'][class_id]['Objects']
#     object_one_id = np.random.choice(sorted(objects.keys()))
#     if find_anno_id:
#       anno_id = '_' + PartnetV0DB.id_index[object_one_id]['anno_id']
#     object_two_id = MeshCorrespondencesDB.get_partnet_id_corr_match(object_one_id + anno_id)
#     return object_one_id, object_two_id

#   def sample_pair(self, class_id):
#     objects_per_class_id = None
#     for objects in self.index:
#       if objects['class'] == class_id:
#         objects_per_class_id = objects
#     if objects_per_class_id is None:
#       raise ValueError(f'missing class_id={class_id}')
#     objects = objects_per_class_id['partnet_data'][class_id]['Objects']

#     object_one_id = np.random.choice(sorted(objects.keys()))
#     if self.is_object_instance_only(class_id):
#       return object_one_id, object_one_id
#     # Go through and find a match in the same group.
#     object_group = objects[object_one_id]['group_label']
#     objects_in_group = []
#     for key in sorted(objects.keys()):
#       if object_group == objects[key]['group_label']:
#         objects_in_group.append(key)
#     object_two_id = np.random.choice(objects_in_group)
#     return object_one_id, object_two_id


# class HouseImages:

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'textures_house'
#     self.local_path = get_local_cache_path('textures_house')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self._cache = {}

#   def sample(self):
#     idx = np.random.randint(len(self.index))
#     id_ = self.index[idx]['id']
#     return self.get_sample(id_)

#   def get_sample(self, id_):
#     if id_ not in self._cache:
#       logging.info(f'TexturesHouse: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_)
#     return self._cache[id_]

#   def _get_sample(self, id_):
#     local_jpg_path = self.local_path / f'{id_}.jpg'
#     if local_jpg_path.exists():
#       return local_jpg_path
#     self._download_sample_jpg(id_, local_jpg_path)
#     return local_jpg_path

#   def _download_sample_jpg(self, id_, local_path):
#     s3_key = f'simnet/textures_house/images/{id_}.jpg'
#     download_s3_file(s3_key, local_path)


# class AdversarialImages:

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'textures_adv'
#     self.local_path = get_local_cache_path('textures_adv')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self._cache = {}

#   def prefetch_all(self, executor):
#     for sample in self.index:
#       id_ = sample['id']
#       yield executor.submit(self.get_sample, id_, disable_cache=True)

#   def sample(self):
#     idx = np.random.randint(len(self.index))
#     id_ = self.index[idx]['id']
#     return self.get_sample(id_)

#   def get_sample(self, id_, disable_cache=True):
#     if disable_cache:
#       return self._get_sample(id_)

#     if id_ not in self._cache:
#       logging.info(f'TexturesADV: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_)
#     return self._cache[id_]

#   def _get_sample(self, id_):
#     local_jpg_path = self.local_path / f'{id_}.jpg'
#     if local_jpg_path.exists():
#       return local_jpg_path
#     self._download_sample_jpg(id_, local_jpg_path)
#     return local_jpg_path

#   def _download_sample_jpg(self, id_, local_path):
#     s3_key = f'simnet/textures_adv/images/{id_}.jpg'
#     download_s3_file(s3_key, local_path)


# class TriImages:

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'textures_tri'
#     self.local_path = get_local_cache_path('textures_tri')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self._cache = {}

#   def prefetch_all(self, executor):
#     for sample in self.index:
#       id_ = sample['id']
#       yield executor.submit(self.get_sample, id_, disable_cache=True)

#   def sample(self):
#     idx = np.random.randint(len(self.index))
#     id_ = self.index[idx]['id']
#     return self.get_sample(id_)

#   def get_sample(self, id_, disable_cache=False):
#     if disable_cache:
#       return self._get_sample(id_)
#     if id_ not in self._cache:
#       # logging.info(f'TexturesTRI: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_)
#     return self._cache[id_]

#   def _get_sample(self, id_):
#     local_sample_path = self.local_path / id_
#     jpg_path = local_sample_path / f'{id_}.jpg'
#     if jpg_path.exists():
#       return jpg_path
#     local_tarfile_path = self._download_sample_tarfile(id_)
#     extract_compressed_tarfile(local_tarfile_path, local_sample_path)
#     return jpg_path

#   def _download_sample_tarfile(self, id_):
#     s3_key = f'simnet/textures_tri/tarfiles/{id_}.tar.zst'
#     local_tarfile_path = self.local_path / f'{id_}.tar.zst'
#     download_s3_file(s3_key, local_tarfile_path)
#     return local_tarfile_path


# class NaturalImages:
#   TEST_IDS = [
#       'COCO_val2014_000000387042',
#       'COCO_val2014_000000387074',
#       'COCO_val2014_000000387079',
#       'COCO_val2014_000000387082',
#       'COCO_val2014_000000387098',
#       'COCO_val2014_000000387136',
#   ]

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'natural_images'
#     self.local_path = get_local_cache_path('natural_images')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self.id_index = {meta['id']: meta for meta in self.index}
#     self._cache = {}

#   def prefetch_all(self, executor):
#     for sample in self.index:
#       id_ = sample['id']
#       yield executor.submit(self.get_sample, id_, disable_cache=True)

#   def sample(self):
#     idx = np.random.randint(len(self.index))
#     id_ = self.index[idx]['id']
#     return self.get_sample(id_)

#   def get_sample_meta(self, id_):
#     assert id_ in self.id_index
#     return self.id_index[id_]

#   def get_sample(self, id_, disable_cache=False):
#     if disable_cache:
#       return self._get_sample(id_)
#     if id_ not in self._cache:
#       # logging.info(f'NaturalImages: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_)
#     return self._cache[id_]

#   def _get_sample(self, id_):
#     local_sample_path = self.local_path / id_
#     jpg_path = local_sample_path / f'{id_}.jpg'
#     if jpg_path.exists():
#       return jpg_path
#     local_tarfile_path = self._download_sample_tarfile(id_)
#     extract_compressed_tarfile(local_tarfile_path, local_sample_path)
#     return jpg_path

#   def _download_sample_tarfile(self, id_):
#     # If file is found locally (part of test dataset) use local copy, if not
#     # download from s3
#     test_path = self.test_dataset_path / 'tarfiles' / f'{id_}.tar.zst'
#     if test_path.exists() and not DISABLE_LOCAL_TEST_FILES:
#       return test_path
#     s3_key = f'simnet/natural_images/tarfiles/{id_}.tar.zst'
#     local_tarfile_path = self.local_path / f'{id_}.tar.zst'
#     download_s3_file(s3_key, local_tarfile_path)
#     return local_tarfile_path


# class MeshCorrespondences:
#   TEST_IDS = [
#       '32074e5642bad0e12c16495e79df12c1_32074e5642bad0e12c16495e79df12c1',
#       '32074e5642bad0e12c16495e79df12c1_ed55f39e04668bf9837048966ef3fcb9',
#       'c776a8bcf17367a4eab0008bcf55b93e_c776a8bcf17367a4eab0008bcf55b93e',
#       'c776a8bcf17367a4eab0008bcf55b93e_f5f0988e18d400a69c12d6260da9ac2b',
#       'ed55f39e04668bf9837048966ef3fcb9_32074e5642bad0e12c16495e79df12c1',
#       'ed55f39e04668bf9837048966ef3fcb9_ed55f39e04668bf9837048966ef3fcb9',
#       'f5f0988e18d400a69c12d6260da9ac2b_c776a8bcf17367a4eab0008bcf55b93e',
#       'f5f0988e18d400a69c12d6260da9ac2b_f5f0988e18d400a69c12d6260da9ac2b',
#   ]

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'mesh_correspondences'
#     self.local_path = get_local_cache_path('mesh_correspondences')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self.id_index = {meta['id']: meta for meta in self.index}
#     self.create_partnet_id_a_index()
#     self._cache = {}

#   def create_partnet_id_a_index(self):
#     self.partnet_id_a_index = {}
#     for meta in self.index:
#       if not meta['partnet_model_id_a'] in self.partnet_id_a_index:
#         self.partnet_id_a_index[meta['partnet_model_id_a']] = [meta]
#       else:
#         self.partnet_id_a_index[meta['partnet_model_id_a']].append(meta)

#   def get_partnet_ids_from_id(self, id_):
#     partnet_model_id_a, _, partnet_model_id_b = id_.partition('_')
#     return partnet_model_id_a, partnet_model_id_b

#   def get_id(self, partnet_model_id_a, partnet_model_id_b):
#     return f'{partnet_model_id_a}_{partnet_model_id_b}'

#   def get_partnet_id_corr_match(self, partnet_id_a):
#     partnet_model_id_b = np.random.choice(self.partnet_id_a_index[partnet_id_a]
#                                          )['partnet_model_id_b']
#     return partnet_model_id_b.split('_')[0]

#   def get_sample_meta(self, partnet_model_id_a, partnet_model_id_b):
#     id_ = self.get_id(partnet_model_id_a, partnet_model_id_b)
#     assert id_ in self.id_index
#     return self.id_index[id_]

#   def exists(self, partnet_model_id_a, partnet_model_id_b):
#     id_ = self.get_id(partnet_model_id_a, partnet_model_id_b)
#     return id_ in self.id_index

#   def get_sample(self, partnet_model_id_a, partnet_model_id_b):
#     id_ = self.get_id(partnet_model_id_a, partnet_model_id_b)
#     if id_ not in self._cache:
#       logging.info(f'MeshCorr: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_, partnet_model_id_a, partnet_model_id_b)
#     return self._cache[id_]

#   def _get_sample(self, id_, partnet_model_id_a, partnet_model_id_b):
#     local_sample_path = self.local_path / id_
#     mesh_corr_path = local_sample_path / self._get_corr_file(partnet_model_id_a, partnet_model_id_b)
#     if mesh_corr_path.exists():
#       return mesh_corr_path
#     local_tarfile_path = self._download_sample_tarfile(id_)
#     extract_compressed_tarfile(local_tarfile_path, local_sample_path)
#     return mesh_corr_path

#   def _get_corr_file(self, partnet_model_id_a, partnet_model_id_b):
#     return f'{partnet_model_id_a}___corr___{partnet_model_id_b}.txt'

#   def _download_sample_tarfile(self, id_):
#     # If file is found locally (part of test dataset) use local copy, if not
#     # download from s3
#     test_path = self.test_dataset_path / 'tarfiles' / f'{id_}.tar.zst'
#     if test_path.exists() and not DISABLE_LOCAL_TEST_FILES:
#       return test_path
#     s3_key = f'mesh_correspondences/tarfiles/{id_}.tar.zst'
#     local_tarfile_path = self.local_path / f'{id_}.tar.zst'
#     download_s3_file(s3_key, local_tarfile_path)
#     return local_tarfile_path


# class PartnetV0:
#   TEST_IDS = [
#       'f5f0988e18d400a69c12d6260da9ac2b',
#       'c776a8bcf17367a4eab0008bcf55b93e',
#       '32074e5642bad0e12c16495e79df12c1',
#       'ed55f39e04668bf9837048966ef3fcb9',
#   ]

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'partnetv0'
#     self.local_path = get_local_cache_path('partnetv0')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self.id_index = {meta['model_id']: meta for meta in self.index}
#     self.create_class_index()
#     self._cache = {}

#   def prefetch_all(self, executor):
#     for id_ in self.id_index.keys():

#       def _func():
#         try:
#           self.get_sample(id_, disable_cache=True)
#         except zstd.ZstdError:
#           print('warning: zstd error')
#           pass

#       yield executor.submit(_func)

#   def create_class_index(self):
#     self.class_index = {}
#     for meta in self.index:
#       if not meta['model_cat'] in self.class_index:
#         self.class_index[meta['model_cat']] = [meta]
#       else:
#         self.class_index[meta['model_cat']].append(meta)

#   def exists(self, model_id):
#     return model_id in self.id_index

#   def get_sample_meta(self, id_):
#     assert id_ in self.id_index
#     return self.id_index[id_]

#   def get_sample(self, id_, disable_cache=False):
#     if disable_cache:
#       return self._get_sample(id_)

#     if id_ not in self._cache:
#       logging.info(f'PartnetV0: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_)
#     return self._cache[id_]

#   def sample_pair(self, class_id):
#     models_in_class = self.class_index[class_id]
#     model_id_one = np.random.choice(models_in_class)['model_id']
#     model_id_two = np.random.choice(models_in_class)['model_id']
#     return model_id_one, model_id_two

#   def sample(self, class_id):
#     models_in_class = self.class_index[class_id]
#     model_id = np.random.choice(models_in_class)['model_id']
#     return model_id

#   def _get_sample(self, id_):
#     local_sample_path = self.local_path / id_
#     if local_sample_path.exists():
#       return local_sample_path
#     local_tarfile_path = self._download_sample_tarfile(id_)
#     extract_compressed_tarfile(local_tarfile_path, local_sample_path)
#     return local_sample_path

#   def _download_sample_tarfile(self, id_):
#     # If file is found locally (part of test dataset) use local copy, if not
#     # download from s3
#     test_path = self.test_dataset_path / 'tarfiles' / f'{id_}.tar.zst'
#     if test_path.exists() and not DISABLE_LOCAL_TEST_FILES:
#       return test_path
#     # Check the local path as well.
#     s3_key = f'partnetv0/tarfiles/{id_}.tar.zst'
#     local_tarfile_path = self.local_path / f'{id_}.tar.zst'
#     download_s3_file(s3_key, local_tarfile_path)
#     return local_tarfile_path


# class ShapeNet:

#   def __init__(self):
#     self.collection = shapenetcore.ShapeNetCoreCollection()
#     # Fetch existing index from cache/s3
#     self.collection.index.get()

#   def sample(self, class_name):
#     model = self.collection.index.search_and_sample('shapenetcore.category', class_name)
#     obj_path = model.model.path() / 'model.obj'
#     assert obj_path.exists()
#     return obj_path

#   def load_id(self, id):
#     row_hash = self.collection.index.search("shapenetcore.id", id)
#     assert len(set(row_hash)) == 1, f"Found {len(row_hash)} entries for {id = } {row_hash}"
#     row_hash = row_hash[0]
#     model = self.collection.index.retrieve(row_hash)
#     obj_path = model.model.path() / 'model.obj'
#     assert obj_path.exists()
#     return obj_path

#   def prefetch_all(self, executor):
#     for model in self.collection.index:
#       try:
#         obj_path = model.model.path() / 'model.obj'
#         obj_path.exists()
#       except ValueError:
#         print('key error, skipping')
#       except zstd.ZstdError:
#         print('warning: zstd error')
#         pass
#       yield executor.submit(obj_path.exists)


# class AdobeV0:
#   TEST_IDS = [
#       '44d2090c0baeb9b272152988eac576ab',
#       '2dd2a5154bcf52c2aeebd3372dbf7ae6',
#   ]

#   def __init__(self):
#     self.test_dataset_path = get_datasets_path() / 'adobev0'
#     self.local_path = get_local_cache_path('adobev0')
#     self.local_path.mkdir(parents=True, exist_ok=True)
#     self.index = read_compressed_json(self.test_dataset_path / 'index.json.zst')
#     self.id_index = {meta['model_id']: meta for meta in self.index}
#     self._cache = {}

#   def prefetch_all(self, executor):
#     for sample in self.index:
#       id_ = sample['model_id']
#       yield executor.submit(self.get_sample, id_, disable_cache=True)

#   def exists(self, model_id):
#     return model_id in self.id_index

#   def get_sample_meta(self, id_):
#     assert id_ in self.id_index
#     return self.id_index[id_]

#   def get_sample(self, id_, disable_cache=False):
#     if disable_cache:
#       return self._get_sample(id_)
#     if id_ not in self._cache:
#       logging.info(f'AdobeV0: Fetching sample {id_}')
#       self._cache[id_] = self._get_sample(id_)
#     return self._cache[id_]

#   def _get_sample(self, id_):
#     local_sample_path = self.local_path / id_
#     if local_sample_path.exists():
#       return local_sample_path
#     local_tarfile_path = self._download_sample_tarfile(id_)
#     extract_compressed_tarfile(local_tarfile_path, local_sample_path)
#     return local_sample_path

#   def _download_sample_tarfile(self, id_):
#     # If file is found locally (part of test dataset) use local copy, if not
#     # download from s3
#     test_path = self.test_dataset_path / 'tarfiles' / f'{id_}.tar.zst'
#     if test_path.exists() and not DISABLE_LOCAL_TEST_FILES:
#       return test_path
#     # Check the local path as well.
#     s3_key = f'adobev0/tarfiles/{id_}.tar.zst'
#     local_tarfile_path = self.local_path / f'{id_}.tar.zst'
#     download_s3_file(s3_key, local_tarfile_path)
#     return local_tarfile_path


# def extract_tarfile(tarfile_path, dst_dir):
#   with open(tarfile_path, 'rb') as raw_fh:
#     tarfile_buf = raw_fh.read()

#   with io.BytesIO(tarfile_buf) as raw_fh:
#     with tarfile.TarFile(fileobj=raw_fh) as tar:
#       members = tar.getmembers()
#       for member in members:
#         if not member.isfile():
#           continue
#         data = tar.extractfile(member).read()
#         assert member.name[0] != '/'
#         member_path = dst_dir / member.path
#         parent_dir = member_path.parent
#         parent_dir.mkdir(parents=True, exist_ok=True)
#         with open(member_path, 'wb') as f:
#           f.write(data)


# def get_datasets_path():
#   return pathlib.Path(__file__).parent / 'datasets'


# def read_compressed_json(path):
#   cctx = zstd.ZstdDecompressor()
#   with open(path, 'rb') as raw_fh:
#     with cctx.stream_reader(raw_fh) as zst_fh:
#       bytes_ = zst_fh.read()
#       str_ = bytes_.decode()
#       x = json.loads(str_, object_pairs_hook=collections.OrderedDict)
#       return x


# def download_s3_file(key, output_path):
#   logging.warning(f'Downloading: {key}')
#   s3 = boto3.client('s3')
#   with open(output_path, 'wb') as fh:
#     s3.download_fileobj(S3_BUCKET, key, fh)


# class TShirts:

#   def __init__(self):
#     self.dataset_names = ["fold1/kopo", "fold2/kopo", "fold3/kopo", "flat_parent_meshes"]
#     self.dataset_paths = [get_datasets_path() / "cloth" / d for d in self.dataset_names]
#     [dp.mkdir(parents=True, exist_ok=True) for dp in self.dataset_paths]

#     self.caches = [set(os.listdir(dp)) for dp in self.dataset_paths]
#     self.cloud_paths = ["simnet/output/cloth/%s" % dn for dn in self.dataset_names]
#     self.key_vertex_infos = []
#     for i, cache in enumerate(self.caches):
#       if not "index.pkl" in cache:
#         download_s3_file("%s/index.pkl" % self.cloud_paths[i], self.dataset_paths[i] / "index.pkl")
#       if not "shirt_data.json" in cache:
#         download_s3_file(
#             "%s/shirt_data.json" % (self.cloud_paths[i]), self.dataset_paths[i] / "shirt_data.json"
#         )
#       with open(str(self.dataset_paths[i] / "shirt_data.json"), 'r') as data_file:
#         self.key_vertex_infos.append(json.loads(json.load(data_file)))
#     self.filenames = [pickle.load(open(str(dp / "index.pkl"), "rb")) for dp in self.dataset_paths]
#     self.multimesh = True

#   def sample(self):
#     dataset_idx = np.random.randint(0, len(self.dataset_names))
#     filename = random.choice(self.filenames[dataset_idx])
#     if "zst" in filename:
#       obj_filename = "%s.obj" % (filename.split(".")[0])
#     else:
#       obj_filename = filename
#     if filename not in self.caches[dataset_idx]:
#       download_s3_file(
#           "%s/%s" % (self.cloud_paths[dataset_idx], filename),
#           self.dataset_paths[dataset_idx] / filename
#       )
#       if "zst" in filename:
#         dctx = zstd.ZstdDecompressor()
#         with open(str(
#             self.dataset_paths[dataset_idx] / filename
#         ), 'rb') as ifh, open(str(self.dataset_paths[dataset_idx] / obj_filename), 'wb') as ofh:
#           dctx.copy_stream(ifh, ofh)
#       self.caches[dataset_idx].add(filename)
#     return self.dataset_paths[dataset_idx] / obj_filename

#   def get_parent_id(self, filename):
#     if not self.multimesh:
#       return "0"
#     dataset_idx = self.dataset_paths.index(filename.parent)
#     return dataset_idx, filename.name.split(".")[0]

#   def get_key_vertices(self, filename):
#     dataset_idx, parent_id = self.get_parent_id(filename)
#     kv = self.key_vertex_infos[dataset_idx][parent_id]
#     return (kv['LEFT_SLEEVE'],
#             kv['RIGHT_SLEEVE']), (kv['NECK'],), (kv['LEFT_CORNER'],
#                                                  kv['RIGHT_CORNER']), (kv['LEFT_VIS_CORNER'],)

#   def get_pick_place(self, filename):
#     dataset_idx, parent_id = self.get_parent_id(filename)
#     kv = self.key_vertex_infos[dataset_idx][parent_id]
#     dataset_name = self.dataset_names[dataset_idx]
#     if dataset_name == "fold1/kopo":
#       return kv['LEFT_CORNER'], kv['RIGHT_CORNER']
#     elif dataset_name == "fold2/kopo":
#       return kv['RIGHT_SLEEVE'], kv['NECK']
#     elif dataset_name == "fold3/kopo":
#       return kv['LEFT_VIS_CORNER'], kv['NECK']
#     else:
#       return kv['LEFT_SLEEVE'], kv['RIGHT_SLEEVE']


# MeshCorrespondencesDB = MeshCorrespondences()
# PartnetV0DB = PartnetV0()
# ShapeNetDB = ShapeNet()
# NaturalImagesDB = NaturalImages()
# TriImagesDB = TriImages()
# AdversarialImagesDB = AdversarialImages()
# HouseImagesDB = HouseImages()
# MMTAnnotationsDB = MMTAnnotations()
# AdobeV0DB = AdobeV0()
# TShirtDB = TShirts()
# PartNetMobilityV0DB = PartNetMobilityV0()


# def _prefetch_parallel(db):
#   with futures.ThreadPoolExecutor(max_workers=40) as executor:
#     all_futures = []
#     for fut in db.prefetch_all(executor=executor):
#       all_futures.append(fut)
#     with tqdm.tqdm(total=len(all_futures)) as pbar:
#       for future in futures.as_completed(all_futures):
#         future.result()
#         pbar.update(1)


# def prefetch_all():
#   _prefetch_parallel(ShapeNetDB)
#   _prefetch_parallel(PartnetV0DB)
#   _prefetch_parallel(AdversarialImagesDB)
#   _prefetch_parallel(AdobeV0DB)
#   _prefetch_parallel(NaturalImagesDB)
#   _prefetch_parallel(TriImagesDB)


# if __name__ == '__main__':
#   main()
