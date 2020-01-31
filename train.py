from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import sys

from tensorflow.python.keras.backend import set_session


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ssr_steps", dest='ssr_steps', type=int, default=0, help="Number of side-step regularization steps after each regular training step.")
parser.add_argument("--epochs", dest='epochs', type=int, default=10, help="Number of training epochs.")

args = parser.parse_args()


batch_size = 50

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

train_size = x_train.shape[0]
iters_per_epoch = train_size // batch_size

val_size = x_val.shape[0]
val_iters = val_size // batch_size



train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.base_model = ResNet50(include_top=False)
    self.dense_1 = Dense(1024, activation='relu')
    self.dense_out = Dense(10, activation='softmax')


  def call(self, x, training=True):
    x = self.base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = self.dense_1(x)
    x = self.dense_out(x)
    return x

model = MyModel()

@tf.function
def proj(u, v):
  return (tf.matmul(tf.transpose(u), v) / tf.matmul(tf.transpose(u), u)) * u

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
      
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def ssr_step(images, labels):
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
      r_o = tf.reshape(r_o, grad.get_shape())
      updated_gradients.append(r_o)
    gradients = updated_gradients
    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



for epoch in range(args.epochs):

  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)
    for i in range(args.ssr_steps):
      ssr_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
  sys.stdout.flush()
