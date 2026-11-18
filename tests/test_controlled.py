"""Deterministic checks for controlled data policies, rates, and optimizer updates."""
import copy
import unittest
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.data_loader import DPDataLoader

from src.baseline import make_model
from src.controlled_data import POLICIES, feature_frame
from src.controlled_training import gaps_from_frame, esr_diagnostics
from src.metrics import binary_metrics
from src.controlled_analysis import aggregate, paired_difference


class ControlledTests(unittest.TestCase):
    def test_poisson_controls_use_the_same_sampling_stream(self):
        dataset = TensorDataset(torch.arange(128).reshape(64, 2).float(), torch.zeros(64))
        generator_a = torch.Generator().manual_seed(777)
        generator_b = torch.Generator().manual_seed(777)
        plain = DPDataLoader.from_data_loader(DataLoader(dataset, batch_size=8, generator=generator_a),
                                               generator=generator_a)
        model = make_model(2, [4, 2])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, clipped = PrivacyEngine(accountant='rdp').make_private(module=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=.05),
                data_loader=DataLoader(dataset, batch_size=8, generator=generator_b),
                noise_multiplier=0., max_grad_norm=1., poisson_sampling=True)
        for (a, _), (b, _) in zip(plain, clipped):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fairness_gaps_are_averaged_before_group_rates(self):
        rows = []
        for seed, accuracies in [(1, [.9, .8]), (2, [.8, .9])]:
            frame = pd.DataFrame({name: accuracies for name in
                                  ['accuracy', 'selection_rate', 'tpr', 'fpr', 'ppv', 'npv']})
            rows.append(dict(feature_policy='include', method='sgd', model_seed=seed,
                             gap=gaps_from_frame(frame)['accuracy_gap']))
        result = aggregate(pd.DataFrame(rows), ['feature_policy', 'method'], ['gap']).iloc[0]
        self.assertAlmostEqual(result['mean'], .1)
        self.assertAlmostEqual(np.mean([.9, .8])-np.mean([.8, .9]), 0.)

    def test_poisson_controls_use_the_same_sampling_stream(self):
        dataset = TensorDataset(torch.arange(128).reshape(64, 2).float(), torch.zeros(64))
        generator_a = torch.Generator().manual_seed(777)
        generator_b = torch.Generator().manual_seed(777)
        plain = DPDataLoader.from_data_loader(DataLoader(dataset, batch_size=8, generator=generator_a),
                                               generator=generator_a)
        model = make_model(2, [4, 2])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, clipped = PrivacyEngine(accountant='rdp').make_private(module=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=.05),
                data_loader=DataLoader(dataset, batch_size=8, generator=generator_b),
                noise_multiplier=0., max_grad_norm=1., poisson_sampling=True)
        for (a, _), (b, _) in zip(plain, clipped):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fairness_gaps_are_averaged_before_group_rates(self):
        rows = []
        for seed, accuracies in [(1, [.9, .8]), (2, [.8, .9])]:
            frame = pd.DataFrame({name: accuracies for name in
                                  ['accuracy', 'selection_rate', 'tpr', 'fpr', 'ppv', 'npv']})
            rows.append(dict(feature_policy='include', method='sgd', model_seed=seed,
                             gap=gaps_from_frame(frame)['accuracy_gap']))
        result = aggregate(pd.DataFrame(rows), ['feature_policy', 'method'], ['gap']).iloc[0]
        self.assertAlmostEqual(result['mean'], .1)
        self.assertAlmostEqual(np.mean([.9, .8])-np.mean([.8, .9]), 0.)

    def test_paired_seed_aggregation_and_sample_sd(self):
        frame = pd.DataFrame([
            dict(feature_policy='include', method='sgd', model_seed=1, accuracy=.9),
            dict(feature_policy='include', method='sgd', model_seed=2, accuracy=.8),
            dict(feature_policy='include', method='dp', model_seed=2, accuracy=.75),
            dict(feature_policy='include', method='dp', model_seed=1, accuracy=.8),
            dict(feature_policy='exclude', method='sgd', model_seed=1, accuracy=.6),
            dict(feature_policy='exclude', method='dp', model_seed=1, accuracy=.55)])
        pairs = paired_difference(frame, ['accuracy'])
        summary = aggregate(pairs, ['feature_policy', 'method'], ['accuracy'])
        include = summary.loc[summary.feature_policy.eq('include')].iloc[0]
        self.assertAlmostEqual(include['mean'], -.075)
        self.assertAlmostEqual(include['std'], np.std([-.1, -.05], ddof=1))
        exclude = summary.loc[summary.feature_policy.eq('exclude')].iloc[0]
        self.assertAlmostEqual(exclude['mean'], -.05)
        self.assertTrue(pd.isna(exclude['std']))
        with self.assertRaises(ValueError):
            paired_difference(frame.loc[~((frame.method == 'sgd') & (frame.model_seed == 2))], ['accuracy'])
        frame['undefined'] = np.nan
        missing = aggregate(frame, ['feature_policy', 'method'], ['undefined'])
        self.assertTrue(missing['mean'].isna().all())
        self.assertTrue(missing.valid_seeds.eq(0).all())

    def test_feature_policies_and_missing_categories(self):
        self.assertEqual(len(POLICIES['include_sensitive']), 16)
        self.assertEqual(len(POLICIES['exclude_sensitive']), 14)
        self.assertTrue({'SEX', 'RAC1P'} <= set(POLICIES['include_sensitive']))
        self.assertFalse({'SEX', 'RAC1P', 'ESR', 'ESP'} & set(POLICIES['exclude_sensitive']))
        result = feature_frame(pd.DataFrame({'AGEP': [20, 30], 'GCL': [1, np.nan], 'SEX': [1, 2]}),
                               ['AGEP', 'GCL', 'SEX'])
        self.assertEqual(result.GCL.tolist(), ['1', 'NOT_APPLICABLE'])
        self.assertEqual(result.SEX.tolist(), ['1', '2'])
        for forbidden in ['ESR', 'ESP', 'Y', 'PWGTP', 'row_id', 'split']:
            with self.assertRaises(ValueError):
                feature_frame(pd.DataFrame(), [forbidden])

    def test_manual_clipping_matches_zero_noise_dp_optimizer(self):
        torch.manual_seed(123)
        original = make_model(3, [4, 2])
        manual = copy.deepcopy(original)
        xb = torch.tensor([[1., 2., -1.], [-2., 1., .5], [.2, -.3, 1.5]])
        yb = torch.tensor([1., 0., 1.])
        c, normalizer, lr = .2, 8, .05
        summed = [torch.zeros_like(p) for p in manual.parameters()]
        for x, y in zip(xb, yb):
            loss = nn.functional.binary_cross_entropy_with_logits(manual(x[None]).squeeze(1), y[None])
            gradients = torch.autograd.grad(loss, list(manual.parameters()))
            norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
            factor = min(1., float(c/(norm+1e-6)))
            for total, gradient in zip(summed, gradients):
                total.add_(gradient*factor)
        for parameter, total in zip(manual.parameters(), summed):
            parameter.data.add_(total, alpha=-lr/normalizer)
        dataset = TensorDataset(torch.zeros(16, 3), torch.zeros(16))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            engine = PrivacyEngine(accountant='rdp')
            wrapped, optimizer, loader = engine.make_private(module=original,
                optimizer=torch.optim.SGD(original.parameters(), lr=lr),
                data_loader=DataLoader(dataset, batch_size=normalizer),
                noise_multiplier=0., max_grad_norm=c, poisson_sampling=True)
            nn.functional.binary_cross_entropy_with_logits(wrapped(xb).squeeze(1), yb).backward()
            optimizer.step()
        self.assertEqual(optimizer.expected_batch_size, normalizer)
        self.assertEqual(loader.sample_rate, .5)
        for expected, actual in zip(manual.parameters(), wrapped.parameters()):
            torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-7)

    def test_unclipped_uses_expected_not_actual_batch(self):
        torch.manual_seed(42)
        a = nn.Linear(2, 1)
        b = copy.deepcopy(a)
        x = torch.tensor([[1., 2.], [3., 4.], [-1., 2.]])
        y = torch.tensor([0., 1., 1.])
        (nn.functional.binary_cross_entropy_with_logits(a(x).squeeze(), y, reduction='sum')/8).backward()
        for index, parameter in enumerate(b.parameters()):
            individual = [torch.autograd.grad(nn.functional.binary_cross_entropy_with_logits(
                          b(x[i:i+1]).squeeze(1), y[i:i+1]), list(b.parameters()), retain_graph=True)[index]
                          for i in range(3)]
            torch.testing.assert_close(list(a.parameters())[index].grad, sum(individual)/8)

    def test_undefined_and_headline_rates(self):
        frame = pd.DataFrame([binary_metrics([0, 1], [0, 1]), binary_metrics([0, 0], [0, 0])])
        self.assertIsNone(gaps_from_frame(frame)['separation'])
        self.assertIsNone(gaps_from_frame(frame)['sufficiency'])
        self.assertAlmostEqual(gaps_from_frame(frame)['independence'], .5)

    def test_esr_false_positive_denominators(self):
        metadata = pd.DataFrame({'Y': [1, 0, 0, 0], 'ESR': [1, 2, 6, 6],
                                 'SEX': [1, 1, 2, 2], 'RAC1P': [1, 1, 2, 2]})
        diagnostics = esr_diagnostics(metadata, np.array([1, 1, 1, 0]))
        female_six = diagnostics.loc[(diagnostics.attribute == 'SEX') &
                       (diagnostics.group == '2') & (diagnostics.ESR == 6)].iloc[0]
        self.assertEqual(female_six.actual_negatives, 2)
        self.assertEqual(female_six.false_positives, 1)
        self.assertEqual(female_six.fpr, .5)
        male_one = diagnostics.loc[(diagnostics.attribute == 'SEX') &
                      (diagnostics.group == '1') & (diagnostics.ESR == 1)].iloc[0]
        self.assertTrue(pd.isna(male_one.fpr))

    def test_epsilon_and_noise_order(self):
        sigmas, epsilons = [], []
        for target in [1., 4., 8.]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sigma = get_noise_multiplier(target_epsilon=target, target_delta=1e-6,
                           sample_rate=.01, steps=100, accountant='rdp', epsilon_tolerance=.001)
                accountant = RDPAccountant()
                accountant.history = [(sigma, .01, 100)]
                achieved = accountant.get_epsilon(1e-6)
            self.assertLessEqual(achieved, target+.001)
            sigmas.append(sigma)
            epsilons.append(achieved)
        self.assertTrue(all(a > b for a, b in zip(sigmas, sigmas[1:])))
        self.assertTrue(all(a < b for a, b in zip(epsilons, epsilons[1:])))


if __name__ == '__main__':
    unittest.main()
