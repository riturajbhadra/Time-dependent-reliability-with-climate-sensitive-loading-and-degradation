"""Checks of probability identities and the retained reference inputs."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np
from scipy.stats import gamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import fit_power_law_degradation_reliability as model
from run_analysis import first_crossing


class ReliabilityChecks(unittest.TestCase):
    def test_reference_integrity(self):
        root = ROOT/'data/reference'
        manifest = json.loads((root/'manifest.json').read_text())
        for name, entry in manifest.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),entry['sha256'])

    def test_nhpp_interval_additivity(self):
        for trend in [0., .0064, -.004]:
            whole = model.nhpp_cumulative(.46,trend,0.,75.)
            parts = model.nhpp_cumulative(.46,trend,0.,30.)+model.nhpp_cumulative(.46,trend,30.,75.)
            self.assertAlmostEqual(float(whole),float(parts),places=11)

    def test_gamma_likelihood_parameterization(self):
        loss = np.array([.001,.005,.008])
        exposure = np.array([.2,.7,1.3])
        c,scale = 7.,.002
        actual = model.gamma_increment_loglik(np.log([c,scale]),loss,exposure)
        expected = gamma.logpdf(loss,a=c*exposure,scale=scale).sum()
        self.assertAlmostEqual(actual,float(expected),places=11)

    def test_preinitiation_and_capacity_bound(self):
        time = np.arange(6,dtype=float)
        loss = model.power_law_loss(time,time,2.,.5,1.)
        np.testing.assert_array_equal(loss[:3],0.)
        np.testing.assert_allclose(loss[3:],[.5,1.,1.])

    def test_service_life_interpolation(self):
        self.assertEqual(first_crossing(np.array([1,2,3]),np.array([4.,3.5,2.5]),3.),2.5)
        self.assertTrue(np.isnan(first_crossing(np.array([1,2]),np.array([4.,3.5]),3.)))


if __name__ == '__main__':
    unittest.main()
