import unittest
import numpy as np
import pandas as pd
from src.metrics import binary_metrics, evaluate


class MetricsTest(unittest.TestCase):
    def test_hand_computed_disparities(self):
        # Group 1 is perfect. Group 2 has TP=1, FN=1, FP=2, TN=0.
        metadata = pd.DataFrame({'Y':[1,1,0,0]*2,'SEX':[1]*4+[2]*4,'RAC1P':[1]*8})
        overall,groups,gaps = evaluate(metadata,np.array([1,1,0,0,0,1,1,1]))
        self.assertAlmostEqual(overall['accuracy'],5/8)
        self.assertAlmostEqual(gaps['SEX']['accuracy_gap'],.75)
        self.assertAlmostEqual(gaps['SEX']['independence'],.25)
        self.assertAlmostEqual(gaps['SEX']['tpr_gap'],.5)
        self.assertAlmostEqual(gaps['SEX']['fpr_gap'],1.)
        self.assertAlmostEqual(gaps['SEX']['separation'],1.)
        self.assertAlmostEqual(gaps['SEX']['ppv_gap'],2/3)
        self.assertAlmostEqual(gaps['SEX']['sufficiency'],1.)
        # Missing race groups must not silently disappear from the full gap.
        self.assertIsNone(gaps['RAC1P']['separation'])

    def test_constant_predictions_and_invalid_alignment(self):
        metrics = binary_metrics([0,1],[0,0])
        self.assertTrue(np.isnan(metrics['ppv']))
        self.assertAlmostEqual(metrics['npv'],.5)
        self.assertTrue(np.isnan(binary_metrics([],[])['accuracy']))
        with self.assertRaises(ValueError): binary_metrics([0,1],[0])


if __name__=='__main__': unittest.main()
