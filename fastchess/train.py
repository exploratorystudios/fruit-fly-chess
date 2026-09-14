"""Train in minutes with bounded CPU threads, validation, and resumable checkpoints."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .features import floats
from .model import export, make_model


def split_games(groups, seed):
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('Need positions from at least two games for independent validation')
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val = np.isin(groups, unique[:max(1, round(len(unique) * .1))])
    return np.flatnonzero(~val), np.flatnonzero(val)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/teacher.npz')
    p.add_argument('--out', default='runs/small')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    p.add_argument('--epochs', type=int, default=20, help='Additional epochs on resume')
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--minutes', type=float, default=10)
    p.add_argument('--patience', type=int, default=5, help='Stop after this many epochs without validation improvement; 0 disables')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    if min(a.epochs, a.batch, a.width, a.threads, a.lr, a.minutes) <= 0:
        p.error('epochs, batch, width, threads, lr and minutes must be positive')
    if a.patience < 0:
        p.error('patience must be nonnegative')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'last.pt'
    if not a.resume and checkpoint.exists():
        p.error(f'{checkpoint} exists; use --resume or another --out')
    with np.load(a.data, allow_pickle=False) as d:
        x, cp, groups = d['X'], d['cp'].astype(np.float32), d['game']
    train_idx, val_idx = split_games(groups, a.seed)
    model = make_model(a.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.0001)
    epoch, best, history = 0, float('inf'), []
    if a.resume:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        if ck['data'] != str(Path(a.data).resolve()) or ck['seed'] != a.seed:
            p.error('Resume requires the same dataset path and split seed; use another output for new data')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        for group in opt.param_groups:
            group['lr'] = a.lr
        epoch, best, history = ck['epoch'], ck['best'], ck['history']
        rng.bit_generator.state = ck['rng']
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters; {device}; '
          f'{len(train_idx):,} train / {len(val_idx):,} validation (separate games)', flush=True)
    target = np.tanh(cp / 400)

    def batch(ids):
        return (torch.from_numpy(floats(x[ids])).to(device),
                torch.from_numpy(target[ids]).to(device))

    @torch.no_grad()
    def validate():
        model.eval()
        loss, base_loss, abs_error, n = 0., 0., 0., 0
        for i in range(0, len(val_idx), a.batch):
            ids = val_idx[i:i+a.batch]
            xb, yb = batch(ids)
            pred = model(xb)
            loss += ((torch.tanh(pred / 400) - yb) ** 2).sum().item()
            base_loss += ((torch.tanh((xb @ model.material) / 400) - yb) ** 2).sum().item()
            # Report centipawn error only for positions with teacher score within +/-1000.
            modest = np.abs(cp[ids]) <= 1000
            abs_error += np.abs(pred.cpu().numpy()[modest] - cp[ids][modest]).sum().item()
            n += modest.sum().item()
        return loss / len(val_idx), base_loss / len(val_idx), abs_error / max(1, n)

    start = time.monotonic()
    stop = False
    stale = 0
    for current in range(epoch + 1, epoch + a.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        seen, total = 0, 0.
        ep_start = time.monotonic()
        try:
            for i in range(0, len(order), a.batch):
                ids = order[i:i+a.batch]
                xb, yb = batch(ids)
                opt.zero_grad(set_to_none=True)
                loss = ((torch.tanh(model(xb) / 400) - yb) ** 2).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                opt.step()
                total += loss.item() * len(ids)
                seen += len(ids)
                if time.monotonic() - start >= a.minutes * 60:
                    stop = True
                    break
        except KeyboardInterrupt:
            stop = True
            print('Interrupted; validating and saving current progress.', flush=True)
        train_seconds = time.monotonic() - ep_start
        val, base, mae = validate()
        row = dict(epoch=current, train_loss=total/max(1, seen), val_loss=val,
                   material_val_loss=base, val_cp_mae=mae, train_seconds=train_seconds,
                   positions_per_second=seen/max(.001, train_seconds))
        history.append(row)
        if val < best:
            best = val
            stale = 0
            export(model, out / 'best.npz')
        else:
            stale += 1
        export(model, out / 'last.npz')
        tmp = checkpoint.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=current,
                        best=best, history=history, width=a.width, seed=a.seed,
                        data=str(Path(a.data).resolve()), rng=rng.bit_generator.state), tmp)
        tmp.replace(checkpoint)
        (out / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f'epoch {current}: train {row["train_loss"]:.4f}, val {val:.4f} '
              f'(material {base:.4f}), MAE {mae:.0f} cp, '
              f'{row["positions_per_second"]:,.0f} positions/s', flush=True)
        if a.patience and stale >= a.patience:
            print('Validation stopped improving; keeping the best regression model. '
                  'Compare playing strength in games before replacing your current engine.')
            stop = True
        if stop:
            break
    print(f'Best validation loss {best:.4f}; playable model: {out / "best.npz"}; '
          f'training/validation/saving took {time.monotonic()-start:.2f}s')


if __name__ == '__main__':
    main()
