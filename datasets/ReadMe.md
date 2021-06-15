-----------------------------------------------------
# dataset 코드 정리
* Dataset을 얻는 utility 입니다.
* 출처: https://github.com/google/uncertainty-baselines/blob/9c29b04dc4500a028ec5b9378af9881fed5f8366/uncertainty_baselines/datasets/datasets.py#L90
## 라이브러리 import 합니다
```
import json
import logging
from typing import Any, List, Tuple, Union
import warnings

import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets.base import BaseDataset
from uncertainty_baselines.datasets.cifar import Cifar100Dataset
from uncertainty_baselines.datasets.cifar import Cifar10CorruptedDataset
from uncertainty_baselines.datasets.cifar import Cifar10Dataset
from uncertainty_baselines.datasets.cifar100_corrupted import Cifar100CorruptedDataset
from uncertainty_baselines.datasets.clinc_intent import ClincIntentDetectionDataset
from uncertainty_baselines.datasets.criteo import CriteoDataset
from uncertainty_baselines.datasets.diabetic_retinopathy_detection import DiabeticRetinopathyDetectionDataset
from uncertainty_baselines.datasets.genomics_ood import GenomicsOodDataset
from uncertainty_baselines.datasets.glue import GlueDatasets
from uncertainty_baselines.datasets.imagenet import ImageNetDataset
from uncertainty_baselines.datasets.mnist import MnistDataset
from uncertainty_baselines.datasets.mnli import MnliDataset
from uncertainty_baselines.datasets.movielens import MovieLensDataset
from uncertainty_baselines.datasets.places import Places365Dataset
from uncertainty_baselines.datasets.random import RandomGaussianImageDataset
from uncertainty_baselines.datasets.random import RandomRademacherImageDataset
from uncertainty_baselines.datasets.svhn import SvhnDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsIdentitiesDataset
from uncertainty_baselines.datasets.toxic_comments import WikipediaToxicityDataset
```
## SpeechCommandsDataset 가 존재하면 import 아니면 에러처리를 합니다
```
try:
  from uncertainty_baselines.datasets.speech_commands import SpeechCommandsDataset  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')
  SpeechCommandsDataset = None
```
## 데이터셋의 이름과 import 주소를 포함한 dictionary 선언합니다
```
DATASETS = {
    'cifar100': Cifar100Dataset,
    'cifar10': Cifar10Dataset,
    'cifar10_corrupted': Cifar10CorruptedDataset,
    'cifar100_corrupted': Cifar100CorruptedDataset,
    'civil_comments': CivilCommentsDataset,
    'civil_comments_identities': CivilCommentsIdentitiesDataset,
    'clinic_intent': ClincIntentDetectionDataset,
    'criteo': CriteoDataset,
    'diabetic_retinopathy_detection': DiabeticRetinopathyDetectionDataset,
    'imagenet': ImageNetDataset,
    'mnist': MnistDataset,
    'mnli': MnliDataset,
    'movielens': MovieLensDataset,
    'places365': Places365Dataset,
    'random_gaussian': RandomGaussianImageDataset,
    'random_rademacher': RandomRademacherImageDataset,
    'speech_commands': SpeechCommandsDataset,
    'svhn_cropped': SvhnDataset,
    'glue/cola': GlueDatasets['glue/cola'],
    'glue/sst2': GlueDatasets['glue/sst2'],
    'glue/mrpc': GlueDatasets['glue/mrpc'],
    'glue/qqp': GlueDatasets['glue/qqp'],
    'glue/qnli': GlueDatasets['glue/qnli'],
    'glue/rte': GlueDatasets['glue/rte'],
    'glue/wnli': GlueDatasets['glue/wnli'],
    'glue/stsb': GlueDatasets['glue/stsb'],
    'wikipedia_toxicity': WikipediaToxicityDataset,
    'genomics_ood': GenomicsOodDataset,
}
```
## 선언되어 있는 데이터셋들의 이름들 반환하는 함수 선언합니다
```
def get_dataset_names() -> List[str]:
  return list(DATASETS.keys())
```
## 이름을 인자로 입력해 dataset builder 클래스를 반환받는 함수 선언합니다
* dataset builder class의 이름과 split(a custom tfds.Split or one of the tfds.Split enums [TRAIN, VALIDAITON, TEST])을 인자로 넣어주면 dataset builder class를 반환합니다.
* 분산 환경에서 작업을 한다면 "distribution_strategy.experimental_distribute_dataset(dataset)"으로 데이터셋을 로드해야 합니다.
```
def get(
    dataset_name: str,
    split: Union[Tuple[str, float], str, tfds.Split],
    **hyperparameters: Any) -> BaseDataset:
    
  hyperparameters_py = {
      k: (v.numpy().tolist() if isinstance(v, tf.Tensor) else v)
      for k, v in hyperparameters.items()
  }
  logging.info(
      'Building dataset %s with additional kwargs:\n%s',
      dataset_name,
      json.dumps(hyperparameters_py, indent=2, sort_keys=True))
  if dataset_name not in DATASETS:
    raise ValueError('Unrecognized dataset name: {!r}'.format(dataset_name))

  dataset_class = DATASETS[dataset_name]
  return dataset_class(
      split=split,
      **hyperparameters)
```

-----------------------------------------------------
# dataset-cifar10 코드 정리
* CIFAR-10 or CIFAR-100  dataset builders를 반환합니다.
* 출처: https://github.com/google/uncertainty-baselines/blob/9c29b04dc4500a028ec5b9378af9881fed5f8366/uncertainty_baselines/datasets/cifar.py

## 라이브러리 import 합니다
```
from typing import Any, Dict, Optional, Union

import numpy as np
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augment_utils
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base
```
## cifar10_mean, cifar10_std의 값을 선언합니다
```
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])
```
## _tuple_dict_fn_converter
```
def _tuple_dict_fn_converter(fn, *args):
  def dict_fn(batch_dict):
    images, labels = fn(*args, batch_dict['features'], batch_dict['labels'])
    return {'features': images, 'labels': labels}
  return dict_fn
```
## Cifar Dataset를 만드는 class를 선언합니다
* name:데이터 세트의 이름
fingerprint_key: fingerprinting 함수를 사용하여 element id를 만드는 데 사용할 문자열을 포함하는 feature의 이름
split: 데이터셋 분할(tfds.Split enums [TRAIN, VALIDAITON, TEST] or 소문자 문자열 이름)
seed: random 시드
validation_percent: validation 세트로 사용할 train 세트의 백분율
shuffle_buffer_size: tf.data.Dataset.shuffle()에서 사용할 example의 수
num_parallel_parser_calls: tf.data.Dataset.map()에서 전처리하는 동안 사용할 병렬 thread 수
drop_remainder: points의 수가 배치 크기와 정확히 같은지, 같지 않을 경우 마지막 데이터 배치를 삭제할지 여부. TPU에서 실행할 경우 이 옵션은 True
normalize: CIFAR 데이터 세트 평균 및 stddev에 의해 각 이미지를 정규화할지 여부
try_gcs: GCS 저장 버전의 데이터 세트 파일 사용 여부
download_data: 로드하기 전에 데이터를 다운로드할지 여부
use_bfloat16: bfloat16 또는 float32에서 데이터를 로드할지 여부
aug_params: data augmentation pre-processing을 위한 hyperparameters
data_dir: 데이터를 읽고 쓰기 위한 디렉토리
is_training: 주어진 split 이 training split인지 아닌지
```
class _CifarDataset(base.BaseDataset):
  def __init__(
      self,
      name: str,
      fingerprint_key: str,
      split: str,
      seed: Optional[Union[int, tf.Tensor]] = None,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      drop_remainder: bool = True,
      normalize: bool = True,
      try_gcs: bool = False,
      download_data: bool = False,
      use_bfloat16: bool = False,
      aug_params: Optional[Dict[str, Any]] = None,
      data_dir: Optional[str] = None,
      is_training: Optional[bool] = None,
      **unused_kwargs: Dict[str, Any]):

    self._normalize = normalize
    dataset_builder = tfds.builder(
        name, try_gcs=try_gcs,
        data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder, validation_percent, split)
    super(_CifarDataset, self).__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        seed=seed,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        fingerprint_key=fingerprint_key,
        download_data=download_data,
        cache=True)

    self._use_bfloat16 = use_bfloat16
    if aug_params is None:
      aug_params = {}
    self._adaptive_mixup = aug_params.get('adaptive_mixup', False)
    ensemble_size = aug_params.get('ensemble_size', 1)
    if self._adaptive_mixup and 'mixup_coeff' not in aug_params:
      # Hard target in the first epoch!
      aug_params['mixup_coeff'] = tf.ones([ensemble_size, 10])
    self._aug_params = aug_params

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: types.Features) -> types.Features:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image_dtype = tf.bfloat16 if self._use_bfloat16 else tf.float32
      use_augmix = self._aug_params.get('augmix', False)
      if self._is_training:
        image_shape = tf.shape(image)
        # Expand the image by 2 pixels, then crop back down to 32x32.
        image = tf.image.resize_with_crop_or_pad(
            image, image_shape[0] + 4, image_shape[1] + 4)
        # Note that self._seed will already be shape (2,), as is required for
        # stateless random ops, and so will per_example_step_seed.
        per_example_step_seed = tf.random.experimental.stateless_fold_in(
            self._seed, example[self._enumerate_id_key])
        # per_example_step_seeds will be of size (num, 3).
        # First for random_crop, second for flip, third optionally for
        # RandAugment, and foruth optionally for Augmix.
        per_example_step_seeds = tf.random.experimental.stateless_split(
            per_example_step_seed, num=4)
        image = tf.image.stateless_random_crop(
            image,
            (image_shape[0], image_shape[0], 3),
            seed=per_example_step_seeds[0])
        image = tf.image.stateless_random_flip_left_right(
            image,
            seed=per_example_step_seeds[1])

        # Only random augment for now.
        if self._aug_params.get('random_augment', False):
          count = self._aug_params['aug_count']
          augment_seeds = tf.random.experimental.stateless_split(
              per_example_step_seeds[2], num=count)
          augmenter = augment_utils.RandAugment()
          augmented = [
              augmenter.distort(image, seed=augment_seeds[c])
              for c in range(count)
          ]
          image = tf.stack(augmented)

        if use_augmix:
          augmenter = augment_utils.RandAugment()
          image = augmix.do_augmix(
              image, self._aug_params, augmenter, image_dtype,
              mean=CIFAR10_MEAN, std=CIFAR10_STD,
              seed=per_example_step_seeds[3])

      # The image has values in the range [0, 1].
      # Optionally normalize by the dataset statistics.
      if not use_augmix:
        if self._normalize:
          image = augmix.normalize_convert_image(
              image, image_dtype, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        else:
          image = tf.image.convert_image_dtype(image, image_dtype)
      parsed_example = example.copy()
      parsed_example['features'] = image

      # Note that labels are always float32, even when images are bfloat16.
      mixup_alpha = self._aug_params.get('mixup_alpha', 0)
      label_smoothing = self._aug_params.get('label_smoothing', 0.)
      should_onehot = mixup_alpha > 0 or label_smoothing > 0
      if should_onehot:
        parsed_example['labels'] = tf.one_hot(
            example['label'], 10, dtype=tf.float32)
      else:
        parsed_example['labels'] = tf.cast(example['label'], tf.float32)

      del parsed_example['image']
      del parsed_example['label']
      return parsed_example

    return _example_parser

  def _create_process_batch_fn(
      self,
      batch_size: int) -> Optional[base.PreProcessFn]:
    if self._is_training and self._aug_params.get('mixup_alpha', 0) > 0:
      if self._adaptive_mixup:
        return _tuple_dict_fn_converter(
            augmix.adaptive_mixup, batch_size, self._aug_params)
      else:
        return _tuple_dict_fn_converter(
            augmix.mixup, batch_size, self._aug_params)
    return None
```
## Cifar10Dataset을 builder class로 생성하는 class 선언합니다
```
class Cifar10Dataset(_CifarDataset):
  def __init__(self, **kwargs):
    super(Cifar10Dataset, self).__init__(
        name='cifar10',
        fingerprint_key='id',
        **kwargs)
```
## Cifar100Dataset을 builder class로 생성하는 class 선언합니다
```
class Cifar100Dataset(_CifarDataset):
  def __init__(self, **kwargs):
    super(Cifar100Dataset, self).__init__(
        name='cifar100',
        fingerprint_key='id',
        **kwargs)
```
