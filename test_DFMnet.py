import time
import tensorflow as tf
import scipy.io as sio
import numpy as np
from scipy.spatial import cKDTree

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_evecs', 120,
                     'number of eigenvectors used for representation')
flags.DEFINE_integer('num_model', 5000, '')
flags.DEFINE_string('test_shapes_dir', './Shapes/', '')
flags.DEFINE_string('files_name', 'tr_reg_', 'name common to all the shapes')
flags.DEFINE_string('log_dir', './Testing/',
                    'directory to save targets results')
flags.DEFINE_string('matches_dir', './Matches/',
                    'directory to matches')


def get_test_pair_source(source_fname):
    input_data = {}
    source_file = '%s%s.mat' % (FLAGS.test_shapes_dir, source_fname)

    # This loads the source but with a target name so next lines re-names
    input_data.update(sio.loadmat(source_file))
    input_data['source_evecs'] = input_data['target_evecs']
    del input_data['target_evecs']
    input_data['source_evecs_trans'] = input_data['target_evecs_trans']
    del input_data['target_evecs_trans']
    input_data['source_shot'] = input_data['target_shot']
    del input_data['target_shot']
    input_data['source_evals'] = np.transpose(input_data['target_evals'])
    del input_data['target_evals']

    return input_data


def get_test_pair_target(target_fname):
    input_data = {}
    target_file = '%s%s.mat' % (FLAGS.test_shapes_dir, target_fname)

    input_data.update(sio.loadmat(target_file))
    input_data['target_evals'] = np.transpose(input_data['target_evals'])

    return input_data


def run_test():
    # Start session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print('restoring graph...')
    saver = tf.train.import_meta_graph('%smodel.ckpt-%s.meta'
                                       % (FLAGS.log_dir, FLAGS.num_model))
    saver.restore(sess, tf.train.latest_checkpoint('%s' % FLAGS.log_dir))
    graph = tf.get_default_graph()

    # Retrieve placeholder variables
    source_evecs = graph.get_tensor_by_name('source_evecs:0')
    source_evecs_trans = graph.get_tensor_by_name('source_evecs_trans:0')
    target_evecs = graph.get_tensor_by_name('target_evecs:0')
    target_evecs_trans = graph.get_tensor_by_name('target_evecs_trans:0')
    source_shot = graph.get_tensor_by_name('source_shot:0')
    target_shot = graph.get_tensor_by_name('target_shot:0')
    phase = graph.get_tensor_by_name('phase:0')
    source_evals = graph.get_tensor_by_name('source_evals:0')
    target_evals = graph.get_tensor_by_name('target_evals:0')

    Ct_est = graph.get_tensor_by_name(
                    'matrix_solve_ls/cholesky_solve/MatrixTriangularSolve_1:0'
                                      )

    for i in range(80, 99):
        input_data_source = get_test_pair_source(FLAGS.files_name + '%.3d' % i)
        source_evecs_ = input_data_source['source_evecs'][:, 0:FLAGS.num_evecs]

        for j in range(i+1, 100):
            t = time.time()

            input_data_target = get_test_pair_target(FLAGS.files_name +
                                                     '%.3d' % j)

            feed_dict = {
                phase: True,
                source_shot: [input_data_source['source_shot']],
                target_shot: [input_data_target['target_shot']],
                source_evecs: [input_data_source['source_evecs'][
                                                        :,
                                                        0:FLAGS.num_evecs
                                                                 ]
                               ],
                source_evecs_trans: [input_data_source[
                                    'source_evecs_trans'
                                                      ][
                                                        0:FLAGS.num_evecs,
                                                        :]
                                         ],
                source_evals: [input_data_source[
                                                'source_evals'
                                                     ][0][0:FLAGS.num_evecs]],
                target_evecs: [input_data_target[
                                                'target_evecs'
                                                     ][:, 0:FLAGS.num_evecs]],
                target_evecs_trans: [input_data_target[
                                       'target_evecs_trans'][
                                                             0:FLAGS.num_evecs,
                                                             :]
                                         ],
                target_evals: [input_data_target[
                                       'target_evals'][0][0:FLAGS.num_evecs]]
                             }

            Ct_est_ = sess.run([Ct_est], feed_dict=feed_dict)
            Ct = np.squeeze(Ct_est_) #Keep transposed

            kdt = cKDTree(np.matmul(source_evecs_, Ct))
            target_evecs_ = input_data_target['target_evecs'][:, 0:FLAGS.num_evecs]

            dist, indices = kdt.query(target_evecs_, n_jobs=-1)
            indices = indices + 1

            print("Computed correspondences for pair: %s, %s." % (i, j) +
                  " Took %f seconds" % (time.time() - t))

            params_to_save = {}
            params_to_save['matches'] = indices
            #params_to_save['C'] = Ct.T
            # For Matlab where index start at 1
            sio.savemat(FLAGS.matches_dir +
                        FLAGS.files_name + '%.3d-' % i +
                        FLAGS.files_name + '%.3d.mat' % j, params_to_save)


def main(_):
    import time
    start_time = time.time()
    run_test()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    tf.app.run()

