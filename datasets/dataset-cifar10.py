from typing import Any, Dict, Optional, Union

import numpy as np
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augment_utils
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])

def _tuple_dict_fn_converter(fn, *args):
  def dict_fn(batch_dict):
    images, labels = fn(*args, batch_dict['features'], batch_dict['labels'])
    return {'features': images, 'labels': labels}
  return dict_fn

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
      aug_params['mixup_coeff'] = tf.ones([ensemble_size, 10])
    self._aug_params = aug_params

  def _create_process_example_fn(self) -> base.PreProcessFn:
    def _example_parser(example: types.Features) -> types.Features:
      #[0, 1]에서 이미지를 반환하는 사전 처리 함수입니다.
      image = example['image']
      image_dtype = tf.bfloat16 if self._use_bfloat16 else tf.float32
      use_augmix = self._aug_params.get('augmix', False)
      if self._is_training:
        image_shape = tf.shape(image)
        # 이미지를 2픽셀 확장한 다음 32x32로 다시 자릅니다.
        image = tf.image.resize_with_crop_or_pad(
            image, image_shape[0] + 4, image_shape[1] + 4)
        per_example_step_seed = tf.random.experimental.stateless_fold_in(
            self._seed, example[self._enumerate_id_key])
        per_example_step_seeds = tf.random.experimental.stateless_split(
            per_example_step_seed, num=4)
        image = tf.image.stateless_random_crop(
            image,
            (image_shape[0], image_shape[0], 3),
            seed=per_example_step_seeds[0])
        image = tf.image.stateless_random_flip_left_right(
            image,
            seed=per_example_step_seeds[1])

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

      # 이미지는 [0, 1] 범위의 값입니다. 데이터 통계를 기준으로 정규화합니다.
      if not use_augmix:
        if self._normalize:
          image = augmix.normalize_convert_image(
              image, image_dtype, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        else:
          image = tf.image.convert_image_dtype(image, image_dtype)
      parsed_example = example.copy()
      parsed_example['features'] = image

      # 라벨은 이미지가 bfloat16인 경우에도 항상 float32입니다.
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

class Cifar10Dataset(_CifarDataset):
  def __init__(self, **kwargs):
    super(Cifar10Dataset, self).__init__(
        name='cifar10',
        fingerprint_key='id',
        **kwargs)

class Cifar100Dataset(_CifarDataset):
  def __init__(self, **kwargs):
    super(Cifar100Dataset, self).__init__(
        name='cifar100',
        fingerprint_key='id',
        **kwargs)
