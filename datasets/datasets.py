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

try:
  from uncertainty_baselines.datasets.speech_commands import SpeechCommandsDataset  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')
  SpeechCommandsDataset = None

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

def get_dataset_names() -> List[str]:
  return list(DATASETS.keys())

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
