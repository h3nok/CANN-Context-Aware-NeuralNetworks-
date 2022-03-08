from unittest import TestCase
from deepclo.core.measures.statistical import l1_norm, l2_norm
from deepclo.core.measures.information_theory import entropy, \
    cross_entropy, joint_entropy, kl_divergence, residual_entropy, mutual_information
import numpy as np


class Test(TestCase):
    # input arrays
    patch_1 = np.array([1, 1, 1, 1])
    patch_2 = np.array([1, 1, 1, 1])
    patch_3 = np.array([1, 1, 0, 0])

    def test_l1_norm(self):
        assert (l1_norm(self.patch_1, self.patch_2)) == 0
        assert (l1_norm(self.patch_3, self.patch_2)) == 7

    def test_l2_norm(self):
        assert (l1_norm(self.patch_1, self.patch_2)) == 0
        assert (l2_norm(self.patch_3, self.patch_2)) > 0

    def test_entropy(self):
        assert entropy(patch=self.patch_1) == 0
        assert entropy(patch=self.patch_3) > 0

    def test_cross_entropy(self):
        assert cross_entropy(self.patch_1, self.patch_2) == 0
        assert cross_entropy(self.patch_1, self.patch_3) > 0

    def test_joint_entropy(self):
        assert joint_entropy(self.patch_1, self.patch_2) == 0
        assert joint_entropy(self.patch_1, self.patch_3) > 0

    def test_kl_divergence(self):
        assert kl_divergence(self.patch_1, self.patch_2) == 0
        assert kl_divergence(self.patch_1, self.patch_3) > 0

    def test_residual_entropy(self):
        assert residual_entropy(self.patch_1) == 0
        assert residual_entropy(self.patch_3) > 0

    def test_mutual_information(self):
        assert mutual_information(self.patch_1, self.patch_2) == 0
        print(mutual_information(self.patch_3, self.patch_1))
