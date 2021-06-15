import os
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total batch size is 128. It is '
                   'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', -1,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
flags.DEFINE_string('data_dir', None,
                    'data_dir to be used for tfds dataset construction.'
                    'It is required when training with cloud TPUs')
flags.DEFINE_bool('download_data', False,
                  'Whether to download data locally when initializing a '
                  'dataset.')
flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('per_core_batch_size', 64,
                     'Batch size per TPU core/GPU. The number of new '
                     'datapoints gathered per batch is this number divided by '
                     'ensemble_size (we tile the batch by that # of times).')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')
flags.DEFINE_float('train_proportion', default=1.0,
                   help='only use a proportion of training set.')
flags.register_validator('train_proportion',
                         lambda tp: tp > 0.0 and tp <= 1.0,
                         message='--train_proportion must be in (0, 1].')

flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
