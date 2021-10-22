import unittest

import numpy as np

from rlkit.testing.testing_utils import are_np_arrays_equal, \
    are_np_array_iterables_equal


class NPTestCase(unittest.TestCase):
    """
    Numpy test case, providing useful assert methods.
    """
    def assertNpEqual(self, np_arr1, np_arr2, msg="Numpy arrays not equal."):
        self.assertTrue(are_np_arrays_equal(np_arr1, np_arr2), msg)

    def assertNpAlmostEqual(
            self,
            np_arr1,
            np_arr2,
            msg="Numpy arrays not equal.",
            threshold=1e-5,
    ):
        self.assertTrue(
            are_np_arrays_equal(np_arr1, np_arr2, threshold=threshold),
            msg
        )

    def assertNpNotEqual(self, np_arr1, np_arr2, msg="Numpy arrays equal"):
        self.assertFalse(are_np_arrays_equal(np_arr1, np_arr2), msg)

    def assertNpArraysEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg=None,
    ):
        msg = msg or "Numpy arrays {} and {} are not equal".format(
            np_arrays1,
            np_arrays2,
        )
        self.assertTrue(
            are_np_array_iterables_equal(
                np_arrays1,
                np_arrays2,
            ),
            msg
        )

    # TODO(vpong): see why such a high threshold is needed
    def assertNpArraysAlmostEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are not almost equal.",
            threshold=1e-4,
    ):
        self.assertTrue(
            are_np_array_iterables_equal(
                np_arrays1,
                np_arrays2,
                threshold=threshold,
            ),
            msg
        )

    def assertNpArraysNotEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are equal."
    ):
        self.assertFalse(are_np_array_iterables_equal(np_arrays1, np_arrays2),
                         msg)

    def assertNpArraysNotAlmostEqual(
            self,
            np_arrays1,
            np_arrays2,
            msg="Numpy array lists are equal.",
            threshold=1e-4,
    ):
        self.assertFalse(
            are_np_array_iterables_equal(
                np_arrays1,
                np_arrays2,
                threshold=threshold,
            ),
            msg
        )

    def assertNpArrayConstant(
            self,
            np_array: np.ndarray,
            constant
    ):
        self.assertTrue(
            (np_array == constant).all(),
            msg="Not all values equal {0}".format(constant)
        )