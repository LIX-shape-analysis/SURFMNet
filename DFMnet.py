import numpy as np
import tensorflow as tf

from loss_DFMnet import *

flags = tf.app.flags
FLAGS = flags.FLAGS


def dfmnet_model(
                phase, source_shot, target_shot,
                source_evecs, source_evecs_trans, source_evals,
                target_evecs, target_evecs_trans, target_evals):
    """Build DFMnet model.

    Args:
        phase : train\test.
        source_shot : SHOT descriptor of source shape.
        target_shot : SHOT descriptor of target shape.
        source_evecs : eigenvectors on source shape.
        source_evecs_trans : source shape eigen vectors,
                            transposed with area preservation factor.
        source_evals : eigen values of the source shape.
        target_evecs : eigenvectors on target shape.
        target_evecs_trans : target shape eigen vectors,
                            transposed with area preservation factor.
        target_evals : eigen values of the target shape.
    """

    net = {}

    for i_layer in range(FLAGS.num_layers):
        with tf.variable_scope("layer_%d" % i_layer) as scope:
            if i_layer == 0:
                net['layer_%d_source' % i_layer] = res_layer(
                                        source_shot,
                                        dims_out=int(source_shot.shape[-1]),
                                        scope=scope,
                                        phase=phase
                                                            )
                scope.reuse_variables()
                net['layer_%d_target' % i_layer] = res_layer(
                                        target_shot,
                                        dims_out=int(target_shot.shape[-1]),
                                        scope=scope,
                                        phase=phase
                                                            )
            else:
                net['layer_%d_source' % i_layer] = res_layer(
                                        net['layer_%d_source' % (i_layer - 1)],
                                        dims_out=int(source_shot.shape[-1]),
                                        scope=scope,
                                        phase=phase
                                                            )
                scope.reuse_variables()
                net['layer_%d_target' % i_layer] = res_layer(
                                        net['layer_%d_target' % (i_layer - 1)],
                                        dims_out=int(source_shot.shape[-1]),
                                        scope=scope,
                                        phase=phase
                                                            )

    #  Project output features on the shape Laplacian eigen functions
    layer_C_est = i_layer + 1   # Grab current layer index
    F = net['layer_%d_source' % (layer_C_est - 1)]
    A = tf.matmul(source_evecs_trans, F)
    net['A'] = A
    G = net['layer_%d_target' % (layer_C_est - 1)]
    B = tf.matmul(target_evecs_trans, G)
    net['B'] = B

    #  FM-layer: evaluate C_est
    net['C_est_AB'], safeguard_inverse = solve_ls(A, B)

    net['C_est_BA'], safeguard_inverse = solve_ls(B, A)

    #  Evaluate loss without any ground-truth or geodesic distance matrix
    with tf.variable_scope("func_map_loss"):
        net_loss, E1, E2, E3, E4 = func_map_layer(
                                net['C_est_AB'], net['C_est_BA'],
                                source_evecs, source_evecs_trans, source_evals,
                                target_evecs, target_evecs_trans, target_evals,
                                F, G
                                                )

    tf.summary.scalar('net_loss_Bijectivity', E1)
    tf.summary.scalar('net_loss_Orthogonality', E2)
    tf.summary.scalar('net_loss_LaplacianCommutativity', E3)
    tf.summary.scalar('net_loss_DescriptorCommutativity', E4)
    tf.summary.scalar('net_loss', net_loss)
    merged = tf.summary.merge_all()

    return net_loss, safeguard_inverse, merged, net


def res_layer(x_in, dims_out, scope, phase):
    """A residual layer implementation.

    Args:
        x_in: input descriptor per point (dims = batch_size X #pts X #channels)
        dims_out: num channels in output.
                  Usually the same as input for a standard resnet layer.
        scope: scope name for variable sharing.
        phase: train\test.
    """

    with tf.variable_scope(scope):
        x = tf.contrib.layers.fully_connected(
                                                x_in,
                                                dims_out,
                                                activation_fn=None,
                                                scope='dense_1')
        x = tf.contrib.layers.batch_norm(
                                            x,
                                            center=True,
                                            scale=True,
                                            is_training=phase,
                                            scope='bn_1')
        x = tf.nn.relu(x, 'relu')
        x = tf.contrib.layers.fully_connected(
                                                x,
                                                dims_out,
                                                activation_fn=None,
                                                scope='dense_2')
        x = tf.contrib.layers.batch_norm(
                                            x,
                                            center=True,
                                            scale=True,
                                            is_training=phase,
                                            scope='bn_2')

        # If dims_out change, modify input via linear projection
        # (as suggested in resNet)
        if not x_in.get_shape().as_list()[-1] == dims_out:
            x_in = tf.contrib.layers.fully_connected(
                                                        x_in,
                                                        dims_out,
                                                        activation_fn=None,
                                                        scope='projection')

        x += x_in

        return tf.nn.relu(x)


def solve_ls(A, B):
    """functional maps layer.

    Args:
        A: source descriptors projected onto source shape eigenvectors.
        B: target descriptors projected onto target shape eigenvectors.

    Returns:
        Ct_est: estimated C (transposed), such that CA ~= B
        safeguard_inverse:
    """

    # Transpose input matrices
    At = tf.transpose(A, [0, 2, 1])
    Bt = tf.transpose(B, [0, 2, 1])

    # Solve C via least-squares
    Ct_est = tf.matrix_solve_ls(At, Bt)
    C_est = tf.transpose(Ct_est, [0, 2, 1], name='C_est')

    # Calculate error for safeguarding
    safeguard_inverse = tf.nn.l2_loss(tf.matmul(At, Ct_est) - Bt)
    safeguard_inverse /= tf.to_float(tf.reduce_prod(tf.shape(A)))

    return C_est, safeguard_inverse

