import tensorflow as tf

class AddGaussianNoiseTF:
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # ensure float
        tensor = tf.cast(tensor, tf.float32)

        # 1. Mask for non-zero pixels
        mask = tensor > 0   # boolean mask, same shape as tensor

        # 2. Count number of pixels where mask is True
        num_pixels = tf.reduce_sum(tf.cast(mask, tf.int32))

        # 3. Generate noise only for masked pixels
        noise = tf.random.normal(
            shape=(num_pixels,),
            mean=self.mean,
            stddev=self.std
        )

        # 4. Apply noise to masked pixels
        flat_tensor = tf.reshape(tensor, [-1])
        flat_mask = tf.reshape(mask, [-1])

        # apply noise only to masked positions
        noisy_flat = tf.tensor_scatter_nd_add(
            flat_tensor,
            indices=tf.where(flat_mask),
            updates=noise
        )

        # restore shape
        noisy = tf.reshape(noisy_flat, tf.shape(tensor))

        # 5. Clip to valid range [0,1]
        return tf.clip_by_value(noisy, 0., 1.)
