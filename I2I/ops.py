import tensorflow as tf


def calculate_gram_matrix(matrix):
    # matrix : 4-D tensor (N, C, H, W)
    # flatten_matrix : 3-D tensor (N, C, H*W)shape = tensor.get_shape()
    flatten_matrix = tf.reshape(matrix, (*matrix.shape[:2], -1))
    transposed_flatten_matrix = tf.transpose(flatten_matrix, (0, 2, 1))
    return tf.matmul(flatten_matrix, transposed_flatten_matrix)