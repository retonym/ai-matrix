import tensorflow as tf

def dice(_x, axis=-1, epsilon=0.000000001, name='', data_type = tf.float32):
  with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
    if data_type == tf.bfloat16:
      with tf.compat.v1.tpu.bfloat16_scope():
        # TODO(yunfei): check this alphas, can it be FP32
        alphas = tf.compat.v1.get_variable('alpha'+name, _x.get_shape()[-1],
                          initializer=tf.constant_initializer(0.0),
                          dtype=data_type)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

        # case: train mode (uses stats of the current batch)
        mean = tf.reduce_mean(_x, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)
        x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
        # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
        x_p = tf.sigmoid(x_normed)
        return alphas * (1.0 - x_p) * _x + x_p * _x
    else:
      alphas = tf.compat.v1.get_variable('alpha'+name, _x.get_shape()[-1],
                          initializer=tf.constant_initializer(0.0),
                          dtype=data_type)
      input_shape = list(_x.get_shape())

      reduction_axes = list(range(len(input_shape)))
      del reduction_axes[axis]
      broadcast_shape = [1] * len(input_shape)
      broadcast_shape[axis] = input_shape[axis]

      # case: train mode (uses stats of the current batch)
      mean = tf.reduce_mean(_x, axis=reduction_axes)
      brodcast_mean = tf.reshape(mean, broadcast_shape)
      std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
      std = tf.sqrt(std)
      brodcast_std = tf.reshape(std, broadcast_shape)
      x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
      # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
      x_p = tf.sigmoid(x_normed)
      return alphas * (1.0 - x_p) * _x + x_p * _x

def parametric_relu(_x):
  alphas = tf.compat.v1.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                       dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg
