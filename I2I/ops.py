import tensorflow as tf


def calculate_gram_matrix(matrix, data_format=None):
    # matrix : 4-D tensor (N, C, H, W)
    # flatten_matrix : 3-D tensor (N, C, H*W)
    # gram_matrix : 3-D tensor (N, C, C)
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()
    if data_format == 'channels_last':
        matrix = tf.transpose(matrix, (0, 3, 1, 2))

    flatten_matrix = tf.reshape(matrix, (*matrix.shape[:2], -1))
    transposed_flatten_matrix = tf.transpose(flatten_matrix, (0, 2, 1))
    gram_matrix = tf.matmul(flatten_matrix, transposed_flatten_matrix)

    if data_format == 'channels_last':
        return tf.transpose(gram_matrix, (0, 2, 3, 1))
    return gram_matrix
