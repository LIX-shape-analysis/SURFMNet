import numpy as np
import random
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def penalty_bijectivity(C_est_AB, C_est_BA):
    """Having both functionals maps for two given shapes,
       composition should yield identity.

    Args:
        C_est_AB : estimated functional from source to target.
        C_est_BA : estimated functional from target to source.
    """

    return tf.nn.l2_loss(
                        tf.subtract(tf.matmul(C_est_AB, C_est_BA),
                                    tf.eye(tf.shape(C_est_AB)[1])
                                    )
                        )


def penalty_ortho(C_est):
    """Orthogonal constraint on the functional map
    implying that the underlying map T is area-preserving.

    Args:
        C_est : estimated functional from source to target or vice-versa.
    """

    return tf.nn.l2_loss(
                         tf.subtract(
                             tf.matmul(
                                       tf.transpose(C_est, perm=[0, 2, 1]),
                                       C_est),
                             tf.eye(tf.shape(C_est)[1]))
                        )


def penalty_laplacian_commutativity(C_est, source_evals, target_evals):
    """Loss function using the preservation
    under isometries of the Laplace-Beltrami operator.

    Args:
        C_est : estimated functional map. From source to target or vice-versa.
        source_evals : eigen values of the source shape.
        target_evals : eigen values of the target shape.
    """

    # Quicker and less memory than taking diagonal matrix
    eig1 = tf.einsum('abc,ac->abc', C_est, source_evals)
    eig2 = tf.einsum('ab,abc->abc', target_evals, C_est)

    return tf.nn.l2_loss(tf.subtract(eig2, eig1))


def penalty_desc_commutativity(
        C_est, F, G, source_evecs, source_evecs_trans,
        target_evecs, target_evecs_trans):
    """Descriptors preservation constraint using commutativity
    from Dorian Nogneng's paper : Informative Descriptor Preservation via
    Commutativity for Shape Matching, 2017 EUROGRAPHICS.

    Args:
        C_est: estimated functional map from source to target or vice-versa.
        F : Descriptors on source shape, in full basis.
        G : Descriptors on target shape, in full basis.
        source_evecs : eigen vectors of target shape.
        source_evecs_trans : source shape eigen vectors, transposed with area
                            preservation factor.
        target_evecs : eigen vectors of target shape.
        target_evecs_trans : target shape eigen vectors, transposed with area
                            preservation factor.
    """

    F_trans = tf.transpose(F, perm=[0, 2, 1])  # Columns become rows
    G_trans = tf.transpose(G, perm=[0, 2, 1])
    # Size : [batch, shot_dim, num_vertices]

    percent = 20  # Chosing percent of the total number of descriptors
    num_desc = int(FLAGS.dim_shot*percent/100)
    batch_range = tf.tile(
                        tf.reshape(
                                    tf.range(FLAGS.batch_size, dtype=tf.int32),
                                    shape=[FLAGS.batch_size, 1, 1]),
                        [1, num_desc, 1])
    random = tf.random_uniform(
                            [FLAGS.batch_size, num_desc, 1],
                            minval=0,
                            maxval=tf.shape(F)[1] - 1,
                            dtype=tf.int32)

    indices = tf.concat([batch_range, random], axis=2)

    F_ = tf.gather_nd(F_trans, indices)  # percent% of descriptors chosen
    G_ = tf.gather_nd(G_trans, indices)

    F_expand = tf.expand_dims(F_, 2)
    G_expand = tf.expand_dims(G_, 2)
    # Size : # [batch, num_desc, 1, num_vertices]

    # This is quicker than taking a diagonal matrix for the descriptor
    F_diag_reduce1 = tf.einsum('abcd,ade->abcde', F_expand, source_evecs)
    G_diag_reduce1 = tf.einsum('abcd,ade->abcde', G_expand, target_evecs)
    # Size : [batch, num_desc, 1, num_vertices, num_evecs]

    F_diag_reduce2 = tf.einsum(
                            'afd,abcde->abcfe',
                            source_evecs_trans,
                            F_diag_reduce1)
    G_diag_reduce2 = tf.einsum(
                            'afd,abcde->abcfe',
                            target_evecs_trans,
                            G_diag_reduce1)
    # Size : #[batch, num_desc, 1, num_evecs, num_evecs]

    C_est_expand = tf.expand_dims(tf.expand_dims(C_est, 1), 1)

    C_est_tile = tf.tile(C_est_expand, [1, num_desc, 1, 1, 1])

    term_source = tf.einsum('abcde,abcef->abcdf', C_est_tile, F_diag_reduce2)
    term_target = tf.einsum('abcef,abcfd->abced', G_diag_reduce2, C_est_tile)

    subtract = tf.subtract(term_source, term_target)

    return tf.nn.l2_loss(subtract)


def func_map_layer(
                C_est_AB, C_est_BA,
                source_evecs, source_evecs_trans, source_evals,
                target_evecs, target_evecs_trans, target_evals,
                F, G):
    """Layer of double functional map loss.

    Args:
        C_est_AB : estimated functional map from source to target.
        C_est_BA : estimated functional map from target to source.
        source_evecs : eigen vectors of target shape.
        source_evecs_trans : source shape eigen vectors,
                            transposed with area preservation factor.
        source_evals : eigen values of the source shape.
        target_evecs : eigen vectors of target shape.
        target_evecs_trans : target shape eigen vectors,
                            transposed with area preservation factor.
        target_evals : eigen values of the target shape.
        F : Descriptors on source shape, in full basis.
        G : Descriptors on target shape, in full basis.
    """

    #############
    # PENALTIES #
    #############

    alpha = 10**3  # Bijectivity
    beta = 10**3   # Orthogonality
    gamma = 1      # Laplacian commutativity
    delta = 10**5  # Descriptor preservation via commutativity

    E1 = (penalty_bijectivity(C_est_AB, C_est_BA) +
          penalty_bijectivity(C_est_BA, C_est_AB))/2

    E2 = (penalty_ortho(C_est_AB) + penalty_ortho(C_est_BA))/2

    E3 = (penalty_laplacian_commutativity(
                                        C_est_AB,
                                        source_evals,
                                        target_evals) +
          penalty_laplacian_commutativity(
                                        C_est_BA,
                                        target_evals,
                                        source_evals))/2

    E4 = (penalty_desc_commutativity(
                                C_est_AB,
                                F, G,
                                source_evecs, source_evecs_trans,
                                target_evecs, target_evecs_trans) +
          penalty_desc_commutativity(
                                C_est_BA,
                                G, F,
                                target_evecs, target_evecs_trans,
                                source_evecs, source_evecs_trans))/2
    # This line for E4 can take a lot of memory depending
    # on the batch size chosen and/or the number of descriptors

    ########
    # LOSS #
    ########

    loss = tf.reduce_mean(alpha * E1 + beta * E2 + gamma * E3 + delta * E4)
    loss /= tf.to_float(tf.shape(C_est_AB)[1] * tf.shape(C_est_AB)[0])

    ##########################
    # IMAGES FOR TENSORBOARD #
    ##########################

    C_est_AB = tf.reshape(
        C_est_AB,
        [FLAGS.batch_size, tf.shape(C_est_AB)[1], tf.shape(C_est_AB)[2], 1])
    tf.summary.image("Estimated_FuncMap_AB", C_est_AB, max_outputs=1)

    C_est_BA = tf.reshape(
        C_est_BA,
        [FLAGS.batch_size, tf.shape(C_est_BA)[1], tf.shape(C_est_BA)[2], 1])
    tf.summary.image("Estimated_FuncMap_BA", C_est_BA, max_outputs=1)

    return loss, E1, E2, E3, E4

