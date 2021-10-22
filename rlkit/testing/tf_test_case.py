import numpy as np
import tensorflow as tf

from rlkit.testing.np_test_case import NPTestCase


class TFTestCase(NPTestCase):
    """
    Tensorflow test case, providing useful assert methods and clean default
    session.
    """
    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.get_default_session() or tf.Session()
        self.sess_context = self.sess.as_default()
        self.sess_context.__enter__()

    def tearDown(self):
        self.sess_context.__exit__(None, None, None)
        self.sess.close()

    def assertParamsEqual(self, network1, network2):
        self.assertNpArraysEqual(
            network1.get_param_values(),
            network2.get_param_values(),
            msg="Parameters are not equal.",
        )

    def assertParamsNotEqual(self, network1, network2):
        self.assertNpArraysNotEqual(
            network1.get_param_values(),
            network2.get_param_values(),
            msg="Parameters are equal.",
        )

    def randomize_param_values(self, network):
        for v in network.get_params():
            self.sess.run(
                v.assign(np.random.rand(*v.get_shape()))
            )
