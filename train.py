from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import tensorflow.keras.models as models
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import layers

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




from tensorflow.python.keras.backend import set_session


batch_size = 6

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
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)




loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
nored_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

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

    #self._trainable_weights.extend(self.dense_1.trainable_weights)
    #self._trainable_weights.extend(self.dense_out.trainable_weights)

  def call(self, x, training=True):
    x = self.base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = self.dense_1(x)
    x = self.dense_out(x)
    return x

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10))
#model.add(layers.Activation('softmax'))
#model = MyModel()

model.build((batch_size, 32, 32, 3))


@tf.function
def ce(y_true_tf, y_pred_tf):
  eps = 1e-6
  #return tf.reduce_sum(tf.square(y_true_tf -  y_pred_tf))
  cliped_y_pref_tf = tf.clip_by_value(y_pred_tf, eps, 1-eps)
  return -tf.reduce_sum(y_true_tf * tf.math.log(cliped_y_pref_tf), axis=1)

@tf.function
def proj(u, v):
  return (tf.matmul(tf.transpose(u), v) / tf.matmul(tf.transpose(u), u)) * u

@tf.function
@gin.configurable(blacklist=['images', 'labels'])
def train_step(images, labels, c_grads, dog_lambda=0.0):
  with tf.GradientTape() as tape2:
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
   reg_loss = reg_loss / (batch_size)
  gradients_reg = tape2.gradient(reg_loss, model.trainable_variables)

  
  g = []
  for x, y in zip(gradients, gradients_reg):
      x = tf.reduce_mean(x, axis=0)
      g.append(x+dog_lambda*y)
  
  if False:
    i = 0
    updated_gradients = []
    for grad in gradients:
      var_ref = model.trainable_variables[i].experimental_ref()
      
      flat_grad = tf.reshape(grad, (-1, 1))
      g_o = flat_grad - proj(c_grads[var_ref], flat_grad)
      c_grads[var_ref].assign_add(g_o)

      g_o = tf.reshape(g_o, grad.get_shape())
      updated_gradients.append(g_o)
      i += 1

    gradients = updated_gradients

  optimizer.apply_gradients(zip(g, model.trainable_variables))

  train_loss(tf.reduce_mean(reg_loss))
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

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


@gin.configurable
def trainer(num_epochs=10, ssr_steps=1):

  c_grads = {}

  for var in model.trainable_variables:
    c_grads[var.experimental_ref()] = tf.Variable(tf.ones(tf.reshape(var, (-1, 1)).get_shape()))

  step = 0
  for epoch in range(num_epochs):

    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
      train_step(images, labels, c_grads)
      for i in range(ssr_steps):
        ssr_step(images, labels)
      step += 1
      #if step % 100 == 0:
      #  for var in model.trainable_variables:
      #    c_grads[var.experimental_ref()].assign(tf.ones(tf.reshape(var, (-1, 1)).get_shape()))

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))
    sys.stdout.flush()

def main(argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  use_neptune = "NEPTUNE_API_TOKEN" in os.environ
  if False and use_neptune:
    neptune.init(project_qualified_name="csadrian/dog")
    print(gin.config._CONFIG)
    exp = neptune.create_experiment(params=gin.config._CONFIG, name="exp")
    #for tag in opts['tags'].split(','):
    #  neptune.append_tag(tag)

  trainer()

if __name__ == '__main__':
  app.run(main)