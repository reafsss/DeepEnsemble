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

def _extract_hyperparameter_dictionary():
  flags_as_dict = FLAGS.flag_values_dict()
  hp_keys = ub.models.models.wide_resnet.HP_KEYS
  hps = {k: flags_as_dict[k] for k in hp_keys}
  return hps

def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir) # 로그 정보를 알려줌(출력X)
  tf.random.set_seed(FLAGS.seed)

  data_dir = FLAGS.data_dir

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

  aug_params = {
      'augmix': FLAGS.augmix, 
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth, 
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width, 
  }

  seeds = tf.random.experimental.stateless_split(
      [FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]

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

  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

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

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer) 
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir) 
    initial_epoch = 0 
    if latest_checkpoint: 
      checkpoint.restore(latest_checkpoint) 
      logging.info('Loaded checkpoint %s', latest_checkpoint) #로그 정보를 알려줌(출력X)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

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

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name) # 로그 정보를 알려줌(출력X)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate, #base_learning_rate, 0.1
        'one_minus_momentum': FLAGS.one_minus_momentum, #one_minus_momentum, 0.1
        'l2': FLAGS.l2, #l2, 2e-4
    })

if __name__ == '__main__':
  app.run(main)
