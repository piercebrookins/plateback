import torch
from backend.app.dino_model import SpatialHead

def test_spatial_residual_initially_preserves_v4_logits_and_supports_native_grid():
 torch.manual_seed(191);model=SpatialHead().eval()
 for size in (24,32):
  x=torch.randn(1,384,size,size)
  with torch.no_grad():
   expected=torch.nn.functional.interpolate(model.base(x),size=(size*2,size*2),mode='bilinear',align_corners=False)
   actual=model(x)
  assert actual.shape==(1,5,size*2,size*2)
  assert torch.equal(actual,expected)

def test_spatial_residual_receives_gradients_with_frozen_v4_head():
 model=SpatialHead().eval()
 for p in model.base.parameters():p.requires_grad=False
 model(torch.randn(1,384,6,6)).square().mean().backward()
 assert all(p.grad is None for p in model.base.parameters())
 assert model.refine[-1].weight.grad.abs().sum()>0
