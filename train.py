from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, AveragePooling2D, Conv2D, BatchNormalization, Activation, Flatten
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import sys
import os

import neptune

import gin
import gin.config
import gin.tf


from absl import flags, app

flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


batch_size = 20

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_TRAIN_SAMPLES = 50000
BS_PER_GPU = batch_size


from tensorflow.python.keras.backend import set_session

@tf.function
def preprocess(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


@tf.function
def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y	



def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_dataset = train_dataset.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(batch_size, drop_remainder=True)
test_dataset = test_dataset.map(preprocess).batch(batch_size, drop_remainder=True)


train_size = x_train.shape[0]
iters_per_epoch = train_size // batch_size

#val_size = x_val.shape[0]
#val_iters = val_size // batch_size



step = tf.Variable(0, trainable=False)
boundaries = [100000, 200000]
values = [0.001, 0.0001, 0.00001]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)

# Later, whenever we perform an optimization step, we pass in the step.
learning_rate = learning_rate_fn(step)


loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
nored_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_sup_loss = tf.keras.metrics.Mean(name='train_sup_loss')
train_reg_loss = tf.keras.metrics.Mean(name='train_reg_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_sup_loss = tf.keras.metrics.Mean(name='test_sup_loss')
test_reg_loss = tf.keras.metrics.Mean(name='test_reg_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(fused=False)(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization(fused=False)(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    #activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization(fused=False)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    #activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

@gin.configurable()
def build_model(depth=14, weight_decay=1e-5, version=1):

    input_shape=(32, 32, 3)

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    return model

model = build_model()



@tf.function
def ce(y_true_tf, y_pred_tf):
  eps = 1e-6
  #return tf.reduce_sum(tf.square(y_true_tf -  y_pred_tf))
  cliped_y_pref_tf = tf.clip_by_value(y_pred_tf, eps, 1-eps)
  return -tf.reduce_sum(y_true_tf * tf.math.log(cliped_y_pref_tf), axis=1)

@tf.function
def proj(u, v):
  return (tf.matmul(tf.transpose(u), v) / tf.matmul(tf.transpose(u), u)) * u

_done = False

@tf.function
@gin.configurable(blacklist=['images', 'labels'])
def train_step(images, labels, dog_lambda=0.0):
  with tf.GradientTape() as tape2:
    #tape2.watch(conv_tensors)
    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      #nored_loss = ce(tf.one_hot(labels, 10), tf.nn.softmax(predictions))
      #loss = tf.reduce_sum(nored_loss)
      loss = loss_object(tf.one_hot(labels, 10), tf.nn.softmax(predictions))
    gradients = tape.jacobian(loss, model.trainable_variables, experimental_use_pfor=False)

    reg_loss = 0.0
    for grad in gradients:
      if len(grad.get_shape()) != 5:
        continue
      grad = tf.reshape(grad, (batch_size, -1))
      M = tf.matmul(grad, tf.transpose(grad))
      #reg_loss += tf.reduce_sum(tf.math.abs(grad), axis=[0,1])
      reg_loss += tf.reduce_sum(tf.math.abs(M-tf.eye(M.get_shape()[0])))
      #reg_loss += tf.reduce_mean(tf.math.abs(M * (tf.ones((M.get_shape()[0], M.get_shape()[0]))-tf.eye(M.get_shape()[0]))))

  gradients_reg = tape2.gradient(reg_loss, model.trainable_variables)
  
  g = []
  for x, y in zip(gradients, gradients_reg):
      x = tf.reduce_mean(x, axis=0)
      g.append(x+dog_lambda*y)
  
  optimizer.apply_gradients(zip(g, model.trainable_variables))

  train_reg_loss(tf.reduce_mean(reg_loss))
  train_sup_loss(tf.reduce_mean(loss))
  train_accuracy(labels, predictions)

@tf.function
@gin.configurable(blacklist=['images', 'labels'])
def ssr_step(images, labels, ssr_alpha=0.01):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  
  if True:
    updated_gradients = []
    for grad in gradients:
      flat_grad = tf.reshape(grad, (-1, 1))
      r = tf.random.normal(flat_grad.get_shape())
      r_o = r - proj(flat_grad, r)
      #r_o = r_o * flat_grad
      r_o = tf.reshape(r_o, grad.get_shape())
      updated_gradients.append(ssr_alpha*r_o)
    gradients = updated_gradients
    
  optimizer.apply_gradients(zip(gradients_both, model.trainable_variables))

  train_sup_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_sup_loss(t_loss)
  test_accuracy(labels, predictions)

@gin.configurable
def trainer(num_epochs=10, ssr_steps=1):

  step = 0
  for epoch in range(num_epochs):

    # Reset the metrics at the start of the next epoch
    train_sup_loss.reset_states()
    train_reg_loss.reset_states()
    train_accuracy.reset_states()
    test_sup_loss.reset_states()
    test_reg_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_dataset:
      train_step(images, labels)
      for i in range(ssr_steps):
        ssr_step(images, labels)
      step += 1

    for test_images, test_labels in test_dataset:
      test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,train_sup_loss.result(),
                        train_accuracy.result()*100,
                        test_sup_loss.result(),
                        test_accuracy.result()*100))
    neptune.send_metric('train_accuracy', x=step, y=train_accuracy.result())
    neptune.send_metric('test_accuracy', x=step, y=test_accuracy.result())
    neptune.send_metric('train_sup_loss', x=step, y=train_sup_loss.result())
    neptune.send_metric('train_reg_loss', x=step, y=train_reg_loss.result())
    neptune.send_metric('test_sup_loss', x=step, y=test_sup_loss.result())

    sys.stdout.flush()

def main(argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  use_neptune = "NEPTUNE_API_TOKEN" in os.environ
  if use_neptune:
    neptune.init(project_qualified_name="csadrian/dog")
    print(gin.operative_config_str())
    exp = neptune.create_experiment(params={}, name="exp")
    #for tag in opts['tags'].split(','):
    #  neptune.append_tag(tag)

  trainer()

if __name__ == '__main__':
  app.run(main)