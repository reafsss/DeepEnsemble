------------------------------------------------------------------
# model-wideResidualNetwork-코드정리
* Wide Residual Network 모델
* 출처: https://github.com/google/uncertainty-baselines/blob/9c29b04dc4500a028ec5b9378af9881fed5f8366/uncertainty_baselines/models/wide_resnet.py#L145
## 라이브러리를 import 합니다
```
import functools
from typing import Any, Dict, Iterable, Optional

import tensorflow as tf
```
## 하이퍼 파라미터 인자 선언
## batchNormalization
* tf.keras.layers.BatchNormalization의 새로운 version 함수 선언
* epsilon 과 momentum은 Torch의 기본값을 사용
## Conv2D 층을 생성하는 함수를 선언합니다
```
def Conv2D(filters, seed=None, **kwargs):
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)
```
## 2개의 3x3 Conv2D로 구성된 basic_block을 생성하는 함수를 선언합니다
```
def basic_block(
    inputs: tf.Tensor,
    filters: int,
    strides: int,
    conv_l2: float,
    bn_l2: float,
    seed: int,
    version: int) -> tf.Tensor:
  x = inputs
  y = inputs
  if version == 2:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
    y = tf.keras.layers.Activation('relu')(y)
  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 3)[:, 0]
  y = Conv2D(filters,
             strides=strides,
             seed=seeds[0],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                         gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters,
             strides=1,
             seed=seeds[1],
             kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(y)
  if version == 1:
    y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(bn_l2),
                           gamma_regularizer=tf.keras.regularizers.l2(bn_l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               seed=seeds[2],
               kernel_regularizer=tf.keras.regularizers.l2(conv_l2))(x)
  x = tf.keras.layers.add([x, y])
  if version == 1:
    x = tf.keras.layers.Activation('relu')(x)
  return x
```
## residual block들로 구성된 group을 생성하는 함수를 선언합니다
```
def group(inputs, filters, strides, num_blocks, conv_l2, bn_l2, version, seed):
  seeds = tf.random.experimental.stateless_split(
      [seed, seed + 1], num_blocks)[:, 0]
  x = basic_block(
      inputs,
      filters=filters,
      strides=strides,
      conv_l2=conv_l2,
      bn_l2=bn_l2,
      version=version,
      seed=seeds[0])
  for i in range(num_blocks - 1):
    x = basic_block(
        x,
        filters=filters,
        strides=1,
        conv_l2=conv_l2,
        bn_l2=bn_l2,
        version=version,
        seed=seeds[i + 1])
  return x
```
## dense, conv 및 batch-norm layer에 대한 L2 매개 변수 추출하는 함수를 선언합니다
```
def _parse_hyperparameters(l2: float, hps: Dict[str, float]):
  assert_msg = ('Ambiguous hyperparameter specifications: either l2 or hps '
                'must be provided (received {} and {}).'.format(l2, hps))
  is_specified = lambda h: bool(h) and all(v is not None for v in h.values())
  only_l2_is_specified = l2 is not None and not is_specified(hps)
  only_hps_is_specified = l2 is None and is_specified(hps)
  assert only_l2_is_specified or only_hps_is_specified, assert_msg
  if only_hps_is_specified:
    assert_msg = 'hps must contain the keys {}!={}.'.format(HP_KEYS, hps.keys())
    assert set(hps.keys()).issuperset(HP_KEYS), assert_msg
    return hps
  else:
    return {k: l2 for k in HP_KEYS}
```
## Wide ResNet을 생성하는 함수를 선언합니다
* 세 개의 잔여 블록 그룹을 사용하여 네트워크 매핑 32x32 -> 16x16 -> 8x8 크기의 공간 특성을 갖고 있습니다.
* input_shape: input_shape
* depth: 총 컨볼루션 레이어 수 입니다.
* width_multiplier: 필터의 수에 곱할 정수입니다.
* num_classes: 출력 클래스 수 입니다.
* l2: L2 정규화 계수입니다.
* version: 1= He et al. (2015)의 original ordering, 2= He et al. (2016)의 preactivation ordering
* hps: 초 매개 변수에 대한 세부 사양입니다.
```
def wide_resnet(
    input_shape: Iterable[int],
    depth: int,
    width_multiplier: int,
    num_classes: int,
    l2: float,
    version: int = 2,
    seed: int = 42,
    hps: Optional[Dict[str, float]] = None) -> tf.keras.models.Model:

  l2_reg = tf.keras.regularizers.l2
  hps = _parse_hyperparameters(l2, hps)

  seeds = tf.random.experimental.stateless_split([seed, seed + 1], 5)[:, 0]
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(16,
             strides=1,
             seed=seeds[0],
             kernel_regularizer=l2_reg(hps['input_conv_l2']))(inputs)
  if version == 1:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            conv_l2=hps['group_1_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[1])
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_2_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[2])
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            conv_l2=hps['group_3_conv_l2'],
            bn_l2=hps['bn_l2'],
            version=version,
            seed=seeds[3])
  if version == 2:
    x = BatchNormalization(beta_regularizer=l2_reg(hps['bn_l2']),
                           gamma_regularizer=l2_reg(hps['bn_l2']))(x)
    x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer=tf.keras.initializers.HeNormal(seed=seeds[4]),
      kernel_regularizer=l2_reg(hps['dense_kernel_l2']),
      bias_regularizer=l2_reg(hps['dense_bias_l2']))(x)
  return tf.keras.Model(
      inputs=inputs,
      outputs=x,
      name='wide_resnet-{}-{}'.format(depth, width_multiplier))
```
## Wide ResNet 모델을 생성하는 create_model 함수를 선언합니다
```
def create_model(
    batch_size: Optional[int],
    depth: int,
    width_multiplier: int,
    input_shape: Iterable[int] = (32, 32, 3),
    num_classes: int = 10,
    l2_weight: float = 0.0,
    version: int = 2,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  del batch_size  # unused arg
  return wide_resnet(input_shape=input_shape,
                     depth=depth,
                     width_multiplier=width_multiplier,
                     num_classes=num_classes,
                     l2=l2_weight,
                     version=version)
```
