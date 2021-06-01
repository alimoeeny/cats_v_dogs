import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fnmatch
import time
from datetime import datetime
import logging
from pympler.tracker import SummaryTracker
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import numpy as np
from dataloader import CNDLoader
import google.cloud.logging
import efficientnet.tfkeras
from tensorflow.keras.models import load_model


if __name__ == "__main__":
  print(f"Fun starts now {time.time()}")
  project_id = 'Cats_v_Dogs_EfficieneNet'

  tf.get_logger().setLevel('WARNING')
  tracker = SummaryTracker()

  try:
    client = google.cloud.logging.Client()
    client.get_default_handler()
    client.setup_logging()
  except Exception as e:
    print(f"ERROR setting up google cloud logging: {e} \nMoving on\n")

  msg = f"TensorFlow version {tf.__version__}"
  print(msg)
  logging.info(msg)

  device = None
  if tf.test.is_built_with_cuda:
    devices = tf.config.list_physical_devices('GPU')
    if devices:
      msg = f"We have these devices available, consider using them: {devices}"
      print(msg)
      logging.info(msg)
      tf.config.experimental.set_memory_growth(devices[0], True)
      tf.config.experimental.get_memory_usage('GPU:0')

  wandb.login(key="7e48787d23b800f370d524967c31a4fd8c7fb1a1")
  wandb.init(project=project_id, entity='alimoeeny')

  date_time_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir = "logs/scalars/" + date_time_prefix
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  checkpoint_path = f"training_checkpoints_{project_id}/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  epoch_count = 1 #20 #90 300

  # Create a callback that saves the model's weights
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_freq=1000, save_weights_only=True,verbose=1)

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  IMAGE_SIDE = 500
  BATCH_SIZE = 100
  FEATURE_FILTERS = [120, 150, 240] #[12, 15, 24]
  KERNEL_SIZE = 3
  STRIDES = [(3,3), (3,3), (3,3)]
  OUTPUT_SIZE = 1 # 1-> probabily of being a dog | 2 -> a 2d probability distribution, probability of catness vs probability of dogness?

  features_shape = (BATCH_SIZE, 2560) # 2560 comes from the EfficientNetB7
  input_shape = (BATCH_SIZE, IMAGE_SIDE, IMAGE_SIDE, 3)
  inputs = tf.keras.layers.Input(shape=input_shape[1:], name="raw image input")
  feature_inputs = tf.keras.layers.Input(shape=features_shape[1:], name="EfficientNetB7 features")
  outputs = inputs
  outputs = tf.keras.layers.Conv2D(FEATURE_FILTERS[0], input_shape=input_shape, kernel_size=KERNEL_SIZE, strides=STRIDES[0], padding='same', data_format='channels_last',  dilation_rate=1, activation='relu', use_bias=True, )(outputs)
  outputs = tf.keras.layers.Conv2D(FEATURE_FILTERS[1], kernel_size=KERNEL_SIZE, strides=STRIDES[1], padding='same', data_format='channels_last',  dilation_rate=1, activation='relu', use_bias=True, )(outputs)
  outputs = tf.keras.layers.Conv2D(FEATURE_FILTERS[2], kernel_size=KERNEL_SIZE, strides=STRIDES[2], padding='same', data_format='channels_last',  dilation_rate=1, activation='relu', use_bias=True, )(outputs)
  outputs = tf.keras.layers.Conv2D(FEATURE_FILTERS[2], kernel_size=KERNEL_SIZE, strides=STRIDES[2], padding='same', data_format='channels_last',  dilation_rate=1, activation='relu', use_bias=True, )(outputs)
  outputs = tf.keras.layers.Flatten()(outputs)
  outputs = tf.keras.layers.Concatenate(axis=-1)([outputs, feature_inputs])
  outputs = tf.keras.layers.Dense(outputs.shape[1]+features_shape[1], activation='relu')(outputs)
  outputs = tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid', use_bias=True)(outputs)
  #outputs = tf.keras.layers.Reshape(target_shape=(OUTPUT_SIZE))(outputs)

  model = tf.keras.Model(inputs = [inputs, feature_inputs], outputs=outputs)
  msg = f"Model: {model.summary(line_length=100)}"
  print(msg)
  logging.info(msg)
  #tf.keras.utils.plot_model(model, show_shapes=True)
  model.compile(optimizer=optimizer, loss=loss_func, metrics=['mean_squared_error', 'binary_crossentropy', ])

  train_CnDLoader = CNDLoader(image_side=IMAGE_SIDE, batch_size=BATCH_SIZE, root_dir='catsvdogs/train')
  test_CnDLoader = CNDLoader(image_side=IMAGE_SIDE, batch_size=1, root_dir='catsvdogs/ali_test')

  msg = f"Train dataset size: {len(train_CnDLoader)}"
  print(msg)
  logging.info(msg)
  config = wandb.config
  config.learning_rate = 0.01
  config.update({
    'epoch_count':epoch_count,
    'kernel_size':KERNEL_SIZE,
    'feature_filters':FEATURE_FILTERS,
    'train_dataset_size': len(train_CnDLoader) * BATCH_SIZE,
    'test_dataset_size': len(test_CnDLoader) * 1,
    })

  # Loads the weights
  available_checkpoints = sorted(fnmatch.filter([x for x in os.listdir('.')], checkpoint_path))
  if available_checkpoints:
    model.load_weights(checkpoint_path)

  # loading pretrained model
  pre_proc_model = efficientnet.tfkeras.EfficientNetB7(weights='imagenet', include_top=True)
  pre_proc_model = tf.keras.applications.EfficientNetB7(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling='avg', classes=1000,
    classifier_activation='softmax',
  )

  for counter in range(0, 2000):
    t_534 = time.time()

    data_x, data_y, file_names = train_CnDLoader[counter]
    x = tf.convert_to_tensor(data_x, dtype=tf.float32)
    y = tf.convert_to_tensor(data_y, dtype=tf.float32)
    #injection = y_faces_center_points

    features = pre_proc_model.predict(x)

    training_history = model.fit(x=[x, features], y=y, batch_size=BATCH_SIZE, epochs=(100 if counter ==-1 else epoch_count), shuffle=True, verbose=0,
      callbacks=[
        tensorboard_callback,
        WandbCallback(labels=[], verbose=0, save_model=False, ),
        checkpoint_callback
        ])
    #wandb.log({"labels": labels})

    #print(f"Training History -> {training_history}")
    prediction = model.predict([x, features])
    prediction_performance = 0.0
    prediction_performance = np.sum(y==(prediction.T>=0.5)) / len(x)
    wandb.log({"prediction performance": prediction_performance})

    test_x, test_y, test_file_names = test_CnDLoader[-1]
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.float32)
    test_features = pre_proc_model.predict(test_x)
    test_prediction = model.predict([test_x, test_features])
    test_performance = 0.0
    test_performance = np.sum(test_y==(test_prediction.T>=0.5)) / len(test_x)
    wandb.log({"test performance": test_performance})

    tracker.print_diff()