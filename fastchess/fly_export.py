"""Export the real FlyWire anatomy of the trained sub-connectome for the browser.

Positions are the measured FlyWire coordinates of each neuron, joined by root id,
so the 3D view shows where these cells actually sit in the fly's brain rather than
an invented layout. Synapse endpoints, signs and the learned gains all come from
the trained checkpoint, so what is drawn is the network that actually plays.
"""
import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np


def load_positions(coordinates_csv, flywire_ids):
    """Join root ids to FlyWire coordinates. One position per neuron."""
    import pandas as pd
    table = pd.read_csv(coordinates_csv)
    first = table.drop_duplicates('root_id').set_index('root_id')['position']
    listed = first.reindex(flywire_ids)
    xyz = np.full((len(flywire_ids), 3), np.nan, dtype=np.float64)
    for row, text in enumerate(listed.values):
        if isinstance(text, str):
            xyz[row] = [float(v) for v in re.findall(r'-?\d+', text)[:3]]
    return xyz, int(np.isnan(xyz[:, 0]).sum())


def load_groups(neurons_csv, flywire_ids):
    """Neuropil group per neuron (LO, ME, CA ...), for colouring by brain region."""
    import pandas as pd
    table = pd.read_csv(neurons_csv, usecols=['root_id', 'group'])
    listed = table.drop_duplicates('root_id').set_index('root_id')['group'].reindex(flywire_ids)
    names = [str(v).split('.')[0] if isinstance(v, str) else 'unknown' for v in listed.values]
    order = sorted({n for n in names})
    index = {name: i for i, name in enumerate(order)}
    return np.array([index[n] for n in names], dtype=np.uint8), order


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='../fly-chess/runs/flychess_finisher_ep4.pt')
    p.add_argument('--subgraph', default='../fly-chess/data/subgraph_6000.pkl')
    p.add_argument('--coordinates', default='../flywire-data/coordinates.csv.gz')
    p.add_argument('--neurons', default='../flywire-data/neurons.csv.gz')
    p.add_argument('--out', default='public/fly')
    p.add_argument('--edges', type=int, default=12000, help='Strongest synapses to draw')
    a = p.parse_args(argv)
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    weights = state['state']
    with open(a.subgraph, 'rb') as handle:
        subgraph = pickle.load(handle)
    flywire_ids = np.asarray(subgraph['flywire_ids'], dtype=np.int64)
    count = len(flywire_ids)

    xyz, missing = load_positions(a.coordinates, flywire_ids)
    if missing:
        raise SystemExit(f'{missing} of {count} neurons have no FlyWire coordinate')
    groups, group_names = load_groups(a.neurons, flywire_ids)

    # Centre and scale to a unit-ish box, keeping the brain's real proportions.
    centre = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    scale = float((xyz.max(axis=0) - xyz.min(axis=0)).max()) / 2
    positions = ((xyz - centre) / scale).astype(np.float32)

    post, pre = weights['edge_index'].numpy()
    effective = (weights['edge_sign'] * weights['edge_mag']
                 * weights['edge_gain'].clamp(min=0)).numpy()
    keep = np.argsort(np.abs(effective))[::-1][:min(a.edges, len(effective))]
    keep = keep[np.argsort(-np.abs(effective[keep]))]

    role = np.zeros(count, dtype=np.uint8)
    role[weights['motor_idx'].numpy()] += 2
    role[weights['sensory_idx'].numpy()] += 1      # 1 sensory, 2 motor, 3 both

    blob = b''.join([
        positions.tobytes(),
        role.tobytes(),
        groups.tobytes(),
        pre[keep].astype(np.uint16).tobytes(),
        post[keep].astype(np.uint16).tobytes(),
        np.sign(effective[keep]).astype(np.int8).tobytes(),
        (np.abs(effective[keep]) / (np.abs(effective[keep]).max() or 1)).astype(np.float32).tobytes(),
    ])
    (out / 'anatomy.bin').write_bytes(blob)

    offset = 0
    layout = {}
    for name, length, dtype in (('positions', count * 3, 'float32'), ('role', count, 'uint8'),
                                ('group', count, 'uint8'), ('edgePre', len(keep), 'uint16'),
                                ('edgePost', len(keep), 'uint16'),
                                ('edgeSign', len(keep), 'int8'),
                                ('edgeWeight', len(keep), 'float32')):
        size = {'float32': 4, 'uint16': 2, 'uint8': 1, 'int8': 1}[dtype]
        layout[name] = {'offset': offset, 'length': length, 'dtype': dtype}
        offset += length * size

    (out / 'anatomy.json').write_text(json.dumps({
        'source': 'FlyWire FAFB v783 — complete adult female brain connectome',
        'neurons': count, 'synapsesTotal': int(len(effective)), 'synapsesDrawn': int(len(keep)),
        'timesteps': int(state.get('timesteps', 32)),
        'sensory': int((role & 1).sum()), 'motor': int((role & 2).sum() // 2),
        'valTop1': state.get('val_acc'), 'valTop5': state.get('val_top5'),
        'groups': group_names, 'arrays': layout,
        'boundsNm': {'min': xyz.min(axis=0).tolist(), 'max': xyz.max(axis=0).tolist()},
    }, indent=2) + '\n')
    print(f'{count:,} neurons, {len(keep):,} of {len(effective):,} synapses drawn, '
          f'{len(blob)/1024:.0f} KiB -> {out}/anatomy.bin')
    print(f'groups: {len(group_names)} neuropils, missing coordinates: {missing}')


if __name__ == '__main__':
    main()
