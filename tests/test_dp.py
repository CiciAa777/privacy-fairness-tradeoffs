"""Synthetic checks: clipped gradients, noise and accountant integration."""
import unittest
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from src.baseline import make_model


class DPTest(unittest.TestCase):
    def test_private_step_and_accounting(self):
        torch.manual_seed(1)
        model=make_model(3,[4,2]);ModuleValidator.validate(model,strict=True)
        dataset=TensorDataset(torch.randn(16,3),torch.arange(16).remainder(2).float())
        loader=DataLoader(dataset,batch_size=8)
        engine=PrivacyEngine(accountant='rdp')
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        model,optimizer,loader=engine.make_private(module=model,optimizer=optimizer,data_loader=loader,
            noise_multiplier=1.5,max_grad_norm=.2,poisson_sampling=True)
        self.assertAlmostEqual(loader.sample_rate,.5)
        xb,yb=next(iter(loader))
        loss=nn.BCEWithLogitsLoss()(model(xb).squeeze(1),yb)
        loss.backward()
        self.assertTrue(all(hasattr(p,'grad_sample') for p in model.parameters()))
        optimizer.clip_and_accumulate()
        # Sum of clipped vectors cannot exceed number of records times C.
        norm=torch.cat([p.summed_grad.flatten() for p in model.parameters()]).norm().item()
        self.assertLessEqual(norm,len(yb)*.2+1e-5)
        summed=torch.cat([p.summed_grad.flatten() for p in model.parameters()]).clone()
        optimizer.add_noise()
        noisy=torch.cat([p.grad.flatten() for p in model.parameters()])
        self.assertFalse(torch.allclose(summed,noisy))
        optimizer.scale_grad();optimizer.original_optimizer.step()
        # Use a new ordinary private step to verify accountant hooks advance.
        optimizer.zero_grad()
        loss=nn.BCEWithLogitsLoss()(model(xb).squeeze(1),yb)
        loss.backward();optimizer.step()
        self.assertEqual(sum(entry[2] for entry in engine.accountant.history),1)
        self.assertTrue(np.isfinite(engine.get_epsilon(1e-5)))
        model.eval()
        with torch.no_grad(): self.assertEqual(model(torch.randn(2,3)).shape,(2,1))


if __name__=='__main__': unittest.main()
