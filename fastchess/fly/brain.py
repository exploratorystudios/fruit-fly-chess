"""A chess policy whose hidden layer *is* the fruit fly's connectome.

The fly brain is used as a fixed recurrent spiking substrate:

    board (773) --W_in--> sensory neurons
                          |
                          v
              [ FlyWire subgraph, LIF spiking, T timesteps ]
                          |
                          v
                 motor neurons --W_out--> move logits (4096)

What is *not* learned: which neuron connects to which, and whether a neuron is
excitatory or inhibitory. Both come straight from the connectome and are held
fixed -- training may only rescale the strength of synapses that biology
already put there, via a per-synapse gain initialised at 1.0.

What is learned: the input projection, the output readout, and those gains.

The LIF dynamics and parameters follow Shiu et al. as implemented in
fly-brain/code/run_pytorch.py (same tau, thresholds, alpha synapses, surrogate
gradient); this module re-implements them in a form that backprops through
time without the benchmark harness around it.
"""

import numpy as np
import torch
import torch.nn as nn

# Shiu et al. LIF parameters, in mV / ms.
TAU_SYN = 5.0
TAU_MEM = 20.0
V_REST = -52.0
V_RESET = -52.0
V_THRESH = -45.0
T_REFRAC = 2.2
DT = 0.5   # ms -- coarser than the 0.1ms benchmark step so BPTT stays tractable


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate gradient backward (as in fly-brain)."""

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        return grad_out / (1.0 + (np.pi * v) ** 2)


spike_fn = ATanSpike.apply


class FlyBrainChess(nn.Module):
    def __init__(self, subgraph, n_in=773, n_out=4096, n_sensory=1024,
                 n_motor=1024, timesteps=32, w_scale=0.275, seed=0):
        super().__init__()
        g = subgraph
        n = int(g['n_neurons'])
        self.n_neurons = n
        self.timesteps = timesteps
        self.w_scale = w_scale

        rng = np.random.default_rng(seed)

        # --- fixed connectome topology -----------------------------------
        pre = torch.from_numpy(np.asarray(g['pre'], dtype=np.int64))
        post = torch.from_numpy(np.asarray(g['post'], dtype=np.int64))
        w = torch.from_numpy(np.asarray(g['weight'], dtype=np.float32))
        # Sort edges into coalesced (row-major) order once, here, so every
        # forward pass can hand torch a sparse tensor that is already coalesced
        # -- coalescing 700k edges on every batch is the single most expensive
        # thing in the training loop otherwise.
        order = torch.argsort(post * n + pre)
        pre, post, w = pre[order], post[order], w[order]
        self.register_buffer('edge_index', torch.stack([post, pre]))
        # Normalise measured synapse counts so the recurrent drive starts in a
        # sane dynamic range; sign (Dale's law) is preserved exactly.
        self.register_buffer('edge_sign', torch.sign(w))
        mag = w.abs()
        self.register_buffer('edge_mag', mag / (mag.mean() + 1e-8))
        self.n_synapses = int(pre.numel())

        # --- learnable per-synapse gain ----------------------------------
        self.edge_gain = nn.Parameter(torch.ones(self.n_synapses))

        # --- which neurons see the board / drive the move ----------------
        # Highest-in-degree neurons make the best readout; highest-out-degree
        # neurons propagate input furthest. Picked deterministically.
        indeg = torch.zeros(n).index_add_(0, post, torch.ones_like(w))
        outdeg = torch.zeros(n).index_add_(0, pre, torch.ones_like(w))
        sensory = torch.topk(outdeg, n_sensory).indices
        motor = torch.topk(indeg, n_motor).indices
        self.register_buffer('sensory_idx', sensory)
        self.register_buffer('motor_idx', motor)

        # --- learnable interface -----------------------------------------
        self.w_in = nn.Linear(n_in, n_sensory)
        self.readout = nn.Linear(n_motor, n_out)
        nn.init.normal_(self.w_in.weight, std=0.05)
        nn.init.zeros_(self.w_in.bias)
        nn.init.normal_(self.readout.weight, std=0.02)
        nn.init.zeros_(self.readout.bias)

        self.refrac_steps = max(1, int(round(T_REFRAC / DT)))
        self.syn_decay = float(np.exp(-DT / TAU_SYN))
        self.mem_factor = DT / TAU_MEM

    def _sparse_weights(self):
        """Assemble the connectome as a sparse matrix for this forward pass.

        Values are sign * measured_magnitude * learned_gain, so autograd sends
        gradient straight back to the per-synapse gains while the indices --
        the actual wiring -- stay frozen.
        """
        # Gain is clamped non-negative so it can only scale a synapse, never
        # invert it. Without this, a gain that trains past zero flips an
        # excitatory synapse into an inhibitory one and Dale's law -- the thing
        # we claim to take from the connectome -- quietly stops holding.
        vals = self.edge_sign * self.edge_mag * self.edge_gain.clamp(min=0.0)
        return torch.sparse_coo_tensor(
            self.edge_index, vals, (self.n_neurons, self.n_neurons),
            is_coalesced=True,
        )

    def forward(self, x, return_spikes=False, raster=None):
        """x: (B, 773) board features -> (B, 4096) move logits."""
        B = x.shape[0]
        dev = x.device
        n = self.n_neurons

        drive = self.w_in(x)                       # (B, n_sensory)
        # constant sensory stimulus: same every timestep, so build it once
        stim = torch.zeros(B, n, device=dev).index_add(1, self.sensory_idx, drive)
        W = self._sparse_weights()

        v = torch.full((B, n), V_REST, device=dev)
        g = torch.zeros(B, n, device=dev)
        spikes = torch.zeros(B, n, device=dev)
        refrac = torch.full((B, n), float(self.refrac_steps), device=dev)

        motor_trace = torch.zeros(B, self.motor_idx.numel(), device=dev)
        spike_count = 0.0

        for _step in range(self.timesteps):
            rec = self.w_scale * torch.sparse.mm(W, spikes.t()).t()

            # refractory gating: a neuron that just fired ignores input
            refrac = torch.where(spikes > 0, torch.zeros_like(refrac), refrac + 1)
            gate = (refrac >= self.refrac_steps).float()

            g = g * self.syn_decay + rec * gate
            v = v + stim
            v = v + self.mem_factor * (g - (v - V_REST))

            spikes = spike_fn(v - V_THRESH)
            v = v - ((v - V_RESET) * spikes).detach()
            g = g - (g * spikes).detach()

            motor_trace = motor_trace + spikes[:, self.motor_idx]
            spike_count = spike_count + spikes.mean()
            if raster is not None:
                raster.append(spikes[0].detach().to('cpu', torch.bool))

        rate = motor_trace / self.timesteps
        logits = self.readout(rate)
        if return_spikes:
            return logits, spike_count / self.timesteps
        return logits
