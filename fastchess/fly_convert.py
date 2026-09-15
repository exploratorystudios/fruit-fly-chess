"""Convert the trained fly checkpoint into a PyTorch-free weight file.

Everything the forward pass needs, as plain arrays, so the model can run wherever
NumPy runs — including a serverless function, where PyTorch will not fit.
"""
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(ROOT / 'model' / 'fly.pt'))
    p.add_argument('--out', default=str(ROOT / 'model' / 'fly.npz'))
    p.add_argument('--name', default='FlyWire 6k connectome')
    a = p.parse_args(argv)
    import torch

    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    w = state['state']
    post, pre = w['edge_index'].numpy()
    # The effective synapse strength the model actually uses: measured sign and
    # magnitude, scaled by the learned per-synapse gain.
    value = (w['edge_sign'] * w['edge_mag'] * w['edge_gain'].clamp(min=0)).numpy()
    np.savez_compressed(
        a.out,
        edge_post=post.astype(np.int32), edge_pre=pre.astype(np.int32),
        edge_value=value.astype(np.float32),
        sensory_idx=w['sensory_idx'].numpy().astype(np.int32),
        motor_idx=w['motor_idx'].numpy().astype(np.int32),
        w_in_weight=w['w_in.weight'].numpy().astype(np.float32),
        w_in_bias=w['w_in.bias'].numpy().astype(np.float32),
        readout_weight=w['readout.weight'].numpy().astype(np.float32),
        readout_bias=w['readout.bias'].numpy().astype(np.float32),
        n_neurons=np.int32(int(max(post.max(), pre.max())) + 1),
        timesteps=np.int32(int(state.get('timesteps', 32))),
        w_scale=np.float32(0.275), name=np.str_(a.name),
        val_top1=np.float32(state.get('val_acc', 0.)),
        val_top5=np.float32(state.get('val_top5', 0.)))
    size = Path(a.out).stat().st_size / 1048576
    print(f'{len(value):,} synapses, {int(max(post.max(), pre.max())) + 1:,} neurons '
          f'-> {a.out} ({size:.1f} MiB)')


if __name__ == '__main__':
    main()
