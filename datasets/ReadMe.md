# dataset 코드 정리
-----------------------------------------------------
* Dataset을 얻는 utility 입니다.
* 출처: https://github.com/google/uncertainty-baselines/blob/9c29b04dc4500a028ec5b9378af9881fed5f8366/uncertainty_baselines/datasets/datasets.py#L90
## 라이브러리 import
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


# dataset-cifar10 코드 정리
