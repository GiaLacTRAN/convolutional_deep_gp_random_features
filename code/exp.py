import tensorflow as tf 
def create_mask(batch_size, D, D_init, keep_prob):
    nrf = tf.shape(D)[1]
    ones = tf.ones([batch_size, nrf])
    keep_prob_matrix = tf.multiply(keep_prob, ones)
    D_init_over_D = tf.divide(D_init, D)
    D_init_over_D_tile = tf.tile(D_init_over_D, [batch_size, 1])

    r_u = tf.random_uniform([batch_size, nrf], minval=0, maxval=1.0, dtype=tf.float32)
    mask = tf.cast(tf.where(r_u < keep_prob_matrix, ones, D_init_over_D_tile), tf.float32)
    return D_init_over_D, mask

def create_binary_scaling_vector(d):
    r_u = tf.random_uniform([1, d], minval=0, maxval=1.0, dtype=tf.float32)
    ones = tf.ones([1, d])
    means = tf.multiply(0.5, ones)
    B = tf.cast(tf.where(r_u > means, ones, tf.multiply(-1.0, ones)), tf.float32)
    return B

if __name__ == "__main__":
    batch_size = 5
    nrf = 8
    keep_prob = 0.5
    D_init = tf.Variable(create_binary_scaling_vector(d = nrf), dtype=tf.float32, trainable=False)
    #D = tf.Variable(D_init, dtype=tf.float32)
    D = tf.Variable(tf.random_normal([1, nrf]), dtype=tf.float32)
    D_init_over_D, mask = create_mask(batch_size, D, D_init, keep_prob)

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        [D_init_over_D_eval, mask_eval] = session.run([D_init_over_D, mask])
        print("D_init_over_D:")
        print(D_init_over_D_eval)
        print("Mask:")
        print(mask_eval)