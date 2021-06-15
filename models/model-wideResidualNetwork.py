import functools
from typing import Any, Dict, Iterable, Optional

import tensorflow as tf

HP_KEYS = ('bn_l2', 'input_conv_l2', 'group_1_conv_l2', 'group_2_conv_l2',
           'group_3_conv_l2', 'dense_kernel_l2', 'dense_bias_l2')

BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,
    momentum=0.9)

def Conv2D(filters, seed=None, **kwargs):
  default_kwargs = {
      'kernel_size': 3,
      'padding': 'same',
      'use_bias': False,
      'kernel_initializer': tf.keras.initializers.HeNormal(seed=seed),
  }
  default_kwargs.update(kwargs)
  return tf.keras.layers.Conv2D(filters, **default_kwargs)

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
