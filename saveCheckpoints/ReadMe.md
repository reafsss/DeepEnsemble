# saveCheckpoints.py 코드 정리

* WideResNet28-10 on CIFAR-10 단일 모델 학습과 하이퍼 파라미터 딥 앙상블을 구현할 때 사용하는 체크 포인트를 저장하는데 사용되는 코드입니다.
* 출처: https://github.com/google/uncertainty-baselines/blob/master/baselines/cifar/deterministic.py#L145

## 라이브러리 import 및 flags 선언합니다
* Data Augmentation 할 때 사용되는 값들과 FLAGS.l2의 값이 None일때 사용되는 하이퍼 파라미터의 세부 사양의 값들을 가진 flags 값을 선언합니다.
```
import os
import time
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_float('label_smoothing', 0., 'Label smoothing parameter in [0,1].')
flags.register_validator('label_smoothing',
                         lambda ls: ls >= 0.0 and ls <= 1.0,
                         message='--label_smoothing must be in [0, 1].')
flags.DEFINE_bool('augmix', False,
                  'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer('aug_count', 1,
                     'Number of augmentation operations in AugMix to perform '
                     'on the input image. In the simgle model context, it'
                     'should be 1. In the ensembles context, it should be'
                     'ensemble_size if we perform random_augment only; It'
                     'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth. This corresponds'
                     'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer('augmix_width', 3,
                     'Augmix width. This corresponds to the k in line 5 in the'
                     'Algorithm box in [4].')
flags.DEFINE_float('bn_l2', None, 'L2 reg. coefficient for batch-norm layers.')
flags.DEFINE_float('input_conv_l2', None,
                   'L2 reg. coefficient for the input conv layer.')
flags.DEFINE_float('group_1_conv_l2', None,
                   'L2 reg. coefficient for the 1st group of conv layers.')
flags.DEFINE_float('group_2_conv_l2', None,
                   'L2 reg. coefficient for the 2nd group of conv layers.')
flags.DEFINE_float('group_3_conv_l2', None,
                   'L2 reg. coefficient for the 3rd group of conv layers.')
flags.DEFINE_float('dense_kernel_l2', None,
                   'L2 reg. coefficient for the kernel of the dense layer.')
flags.DEFINE_float('dense_bias_l2', None,
                   'L2 reg. coefficient for the bias of the dense layer.')
flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')
FLAGS = flags.FLAGS
```
## 하이퍼 파라미터를 세팅합니다
* hp_keys = ('bn_l2', 'input_conv_l2', 'group_1_conv_l2', 'group_2_conv_l2','group_3_conv_l2', 'dense_kernel_l2', 'dense_bias_l2')
* hps = {'bn_l2':None, 'input_conv_l2':None, 'group_1_conv_l2':None, 'group_2_conv_l2':None,'group_3_conv_l2':None, 'dense_kernel_l2':None, 'dense_bias_l2':None}
```
def _extract_hyperparameter_dictionary():
  flags_as_dict = FLAGS.flag_values_dict()
  hp_keys = ub.models.models.wide_resnet.HP_KEYS
  hps = {k: flags_as_dict[k] for k in hp_keys}
  return hps
```
## main 함수
### log 값들을 처리하는 로깅 처리기를 선언합니다
* argv 는 사용 안하므로 제거합니다.
```
def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv
```
### output 디렉토리와  global seed를 선언합니다
* output_dir에 따른 디렉토리를 작성하고 global한 random seed를 생성합니다. data_dir이 존재한다면 data_dir을 선언합니다.
```
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir) # 로그 정보를 알려줌(출력X)
  tf.random.set_seed(FLAGS.seed)

  data_dir = FLAGS.data_dir
```
### GPU or TPU 환경을 설정합니다
```
  if FLAGS.use_gpu: #GPU 환경 설정
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else: # TPU 환경 설정
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
```
### 데이터셋 환경설정을 합니다
* tfds.core.DatasetBuilder의 dataset(=cifar10) 가져와서 정보를 저장합니다. batch_size는 64*8인 값으로 설정해주고 train_dataset_size(=50000 * 1.0), steps_per_epoch(=50000/(64*8)), Steps_per_epoch(=10000//(64*8)), num_classes(=10)을 각각 선언해줍니다.
```
  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  train_dataset_size = (
      ds_info.splits['train'].num_examples * FLAGS.train_proportion)
  steps_per_epoch = int(train_dataset_size / batch_size) 
  logging.info('Steps per epoch %s', steps_per_epoch)# 로그 정보를 알려줌(출력X)
  logging.info('Size of the dataset %s', ds_info.splits['train'].num_examples)# 로그 정보를 알려줌(출력X)
  logging.info('Train proportion %s', FLAGS.train_proportion)# 로그 정보를 알려줌(출력X)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size 
  num_classes = ds_info.features['label'].num_classes
```
### aug_params를 선언합니다
* augmix: 입력 데이터에 augmix를 수행할지 여부를 True or False로 입력해줍니다(False)
* aug_count: augmix에서 수행할 증강 작업 수 입니다(1)
* augmix_depth: 표본의 깊이입니다(-1)
* augmix_prob_coeff: augmix 확률계수입니다(0.5)
* augmix_width: augmix 폭입니다(3)
```
  aug_params = {
      'augmix': FLAGS.augmix, 
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth, 
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width, 
  }
```
### seed를 생성합니다
* 2개의 seed를 생성합니다([1769886085, 86449935])
```
  seeds = tf.random.experimental.stateless_split(
      [FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]
```
### 데이터셋을 생성합니다
* datasets.py에 선언되어 있는 get함수를 사용해 train_builder를 생성합니다. 생성된 train_builder 데이터로 train_dataset을 선언합니다. 그 후 validation_dataset을 선언하는데 이 코드에선 trainproportion이 1.0이므로 None의 값을 넣어줍니다. train_dataset을 생성했던 방법과 같이 test_dataset도 생성합니다.
```
  train_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      seed=seeds[0],
      aug_params=aug_params,
      validation_percent=1. - FLAGS.train_proportion,)
  train_dataset = train_builder.load(batch_size=batch_size)
  validation_dataset = None 
  steps_per_validation = 0
  clean_test_builder = ub.datasets.get(
      FLAGS.dataset,
      split=tfds.Split.TEST,
      data_dir=data_dir)
  clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
```
### 분산환경에 맞춰 다시 데이터셋 선언
* 모델을 분산환경에 맞춰 학습시킬 것이므로 데이터셋도 분산환경에 맞춰 다시 선언해줍니다.
```
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
```
### 지정된 디렉토리에 저장될 요약 파일 작성
* 지정된 디렉토리에 저장될 요약 파일을 작성하는 summary_writer 함수를 생성합니다.
```
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))
```
### 여러 장치로 분산시켜 훈련(TPU or GPU)
* 환경설정된 gpu or tpu 환경을 오픈하고 wide_resnet28-10 모델을 구축합니다. learningRate, decay_epochs를 각각 선언해주고 tf.keras.optimizers.schedules.LearningRateScheduler와 같은 역할을 하는 ub.schedules.WarmUpPiecewiseConstantSchedule 함수로 학습속도를 조정해줍니다. 그 후 Gradient descent optimizer를 선언해줍니다. 
```
  with strategy.scope():
    logging.info('Building ResNet model') # 로그 정보를 알려줌(출력X)
    model = ub.models.wide_resnet(
        input_shape=(32, 32, 3),
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        l2=FLAGS.l2,
        hps=_extract_hyperparameter_dictionary(),
        seed=seeds[1])
    logging.info('Model input shape: %s', model.input_shape) # 로그 정보를 알려줌(출력X)
    logging.info('Model output shape: %s', model.output_shape) # 로그 정보를 알려줌(출력X)
    logging.info('Model number of weights: %s', model.count_params()) # 로그 정보를 알려줌(출력X)
    base_lr = FLAGS.base_learning_rate * batch_size / 128 
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200 
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule( 
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule, 
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=True)
```
### 평가지표 행렬
* train 데이터셋의 NLL 
* train 데이터셋의 accuracy 
* train 데이터셋의 loss 
* train 데이터셋의 ece 
* test 데이터셋의 NLL 
* test 데이터셋의 accuracy 
* test 데이터셋의 ece
```
    metrics = { 
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy': 
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': 
            tf.keras.metrics.Mean(),
        'train/ece': 
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
    }
```
### Checkpoint
* checkpoint에 모델과 optimizer 읽을 수 있는 값을 저장합니다. latest_checkpoint에 최근에 저장된 checkpoint 파일의 파일 이름을 저장하고 없으면 false반환하고 초기 epoch 선언하는데 이는 모델을 돌렸을 때 중간에 멈췄다면 이어서 돌릴 수 있게 하기위함입니다. 전에 학습한 체크포인트가 디렉토리 안에 존재한다면 체크포인트를 latest_checkpoint로 update하고 initial_epoch을 latest_checkpoint 이후로 설정합니다.
```
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer) 
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir) 
    initial_epoch = 0 
    if latest_checkpoint: 
      checkpoint.restore(latest_checkpoint) 
      logging.info('Loaded checkpoint %s', latest_checkpoint) #로그 정보를 알려줌(출력X)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
```
### Train
* loss에 대하여 layer의 학습가능한 weight의 gradient 를 저장하며 학습을 수행하는 함수를 선언합니다.
```
  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features'] #변수 선언
      labels = inputs['labels'] #라벨 선언

      with tf.GradientTape() as tape: #레이어가 입력에 적용하는 연산은 gradient Tape에 기록, 자동적으로 미분
        logits = model(images, training=True) # logits 저장
        if FLAGS.label_smoothing == 0.:
          negative_log_likelihood = tf.reduce_mean( #sparse_categorical_crossentropy의 모든 텐서 차원의 평균 계산
              tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                              logits,
                                                              from_logits=True))
        
        l2_loss = sum(model.losses) # l2 로스 계산
        loss = negative_log_likelihood + l2_loss # loss 계산
        scaled_loss = loss / strategy.num_replicas_in_sync #loss 스케일링(장치 수로 나누기(=8))

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables)) #optimizer 를 통해 업데이트 되는 gradient를 사용

      probs = tf.nn.softmax(logits) #softmax activation
      metrics['train/ece'].add_batch(probs, label=labels) #train/ece update
      metrics['train/loss'].update_state(loss) #train/loss update
      metrics['train/negative_log_likelihood'].update_state( #train/negative_log_likelihood update
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits) #train/accuracy update

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)): #올바른 복제본 별 데이터를 단위에 맞춰서 제공
      strategy.run(step_fn, args=(next(iterator),))
```
### Test
* 검증을 수행하는 함수를 선언합니다.
```
  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features'] #변수 선언
      labels = inputs['labels'] #라벨 선언
      logits = model(images, training=False) #logits 저장
      probs = tf.nn.softmax(logits) #softmax activation
      negative_log_likelihood = tf.reduce_mean( # NLL 계산
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        metrics[f'{dataset_split}/negative_log_likelihood'].update_state( #test/negative_log_likelihood update
            negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy'].update_state(labels, probs) #test/accuracy update
        metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels) #test/ece update

    for _ in tf.range(tf.cast(num_steps, tf.int32)): #올바른 복제본 별 데이터를 단위에 맞춰서 제공
      strategy.run(step_fn, args=(next(iterator),))
```
### 학습과 검증을 진행합니다
```
  train_iterator = iter(train_dataset)
  start_time = time.time() #시작 시간 저장
  

  for epoch in range(initial_epoch, FLAGS.train_epochs): #학습, 검증 epoch 진행, (0,200)
    logging.info('Starting to run epoch: %s', epoch)# 로그 정보를 알려줌(출력X)
    train_start_time = time.time()  # 시작 시간 저장
    train_step(train_iterator)  #train-step 함수 시작(학습 진행)
    
    # 시간 계산
    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * FLAGS.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps,
                   epoch + 1,
                   FLAGS.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
    logging.info(message)# 로그 정보를 알려줌(출력X)
    

    datasets_to_evaluate = {'clean': test_datasets['clean']}
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)# 로그 정보를 알려줌(출력X)
      logging.info('Starting to run eval at epoch: %s', epoch)# 로그 정보를 알려줌(출력X)
      test_start_time = time.time() #시작 시간 저장
      test_step(test_iterator, 'test', dataset_name, steps_per_eval) #test-step 함수 시작(검증 진행)

      logging.info('Done with testing on %s', dataset_name)# 로그 정보를 알려줌(출력X)


    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',# 로그 정보를 알려줌(출력X)
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',# 로그 정보를 알려줌(출력X)
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    
    total_results = {name: metric.result() for name, metric in metrics.items()} #전체 reslut 저장
    total_results = { # key value 형태로 저장되어 있는 값들 정렬
        k: (list(v.values())[0] if isinstance(v, dict) else v)
        for k, v in total_results.items()
    }
    with summary_writer.as_default(): # summary 로그 기록
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values(): # metric 초기화
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and #checkpoint_interval 마다 checkpoint 저장, checkpoint_interval=25
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)# 로그 정보를 알려줌(출력X)
```
### final checkpoint, summary 기록
* 모든 학습과 검증이 끝난 후 마지막 체크포인트를 저장하고 summary 로그를 기록합니다.
```
  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name) # 로그 정보를 알려줌(출력X)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate, #base_learning_rate, 0.1
        'one_minus_momentum': FLAGS.one_minus_momentum, #one_minus_momentum, 0.1
        'l2': FLAGS.l2, #l2, 2e-4
    })
```
# main 함수를 실행합니다
```
if __name__ == '__main__':
  app.run(main)
```
