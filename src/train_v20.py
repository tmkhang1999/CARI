"""
Training script for V20 — the final CARI pipeline.

Separate from train_v17.py because V20 is trained DIFFERENTLY:
  • From scratch (V17 resumes from a 19k synthetic checkpoint).
  • 3-phase curriculum on Hypersim → +OpenRooms → +MID (no InteriorVerse).
  • Shading-first 2-head model (IntrinsicDecompositionV20).
  • Phase-end checkpoints are PERMANENTLY kept (the 2-checkpoint prune rule only
    applies to intra-phase checkpoints) so each curriculum stage is recoverable.

All heavy machinery (model build, losses, train step, validation, checkpoint IO)
is imported from train_v17 — only the orchestration / curriculum / checkpoint
retention differs here.
"""

import os
import gc
import sys
import math
from pathlib import Path

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Reuse the shared training machinery verbatim (version-agnostic).
from train_v17 import (
    parse_args,
    load_config,
    build_stage1_model,
    build_optimizer_stage1,
    train_one_step,
    validate,
    save_checkpoint,
    load_checkpoint,
    _resolve_resume_path,
    _extract_iter_from_name,
    _log_ordered_scalars,
    _log_dataset_examples,
)
from losses.flexible_loss_v17 import V17Loss
from data.hypersim_dataset import get_hypersim_loader
from data.mixed_dataset import get_mixed_loader


def _install_signal_logger():
    """Log the signal + parent context on SIGTERM/SIGHUP so an external kill (pkill, a
    scheduler/watchdog, a terminal disconnect) is identifiable next time the run dies with
    "Terminated". Python's handler cannot see the SENDER pid, but the parent process and its
    cmdline cover the common cases (shell, tmux, cron wrapper, job manager). Exits 128+signum
    (143 for SIGTERM) — a diagnostic exit, not a graceful checkpoint (periodic ckpts already
    cover state). SIGINT (Ctrl-C) is left to Python's default so manual interrupts are unaffected."""
    import signal
    import datetime

    def _handler(signum, frame):
        name = signal.Signals(signum).name
        ppid = os.getppid()
        try:
            with open(f'/proc/{ppid}/cmdline', 'rb') as f:
                pcmd = f.read().replace(b'\x00', b' ').decode(errors='replace').strip()
        except Exception:
            pcmd = '<unknown>'
        print(f"\n[{datetime.datetime.now().isoformat(timespec='seconds')}] "
              f"RECEIVED {name} (sig {signum}) — pid={os.getpid()} ppid={ppid} "
              f"parent=[{pcmd}]", flush=True)
        os._exit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _handler)
        except (ValueError, OSError):
            pass  # e.g. not in the main thread; best-effort only


def main():
    _install_signal_logger()
    args = parse_args()
    config = load_config(args.config, args.version)
    print(yaml.dump(config, default_flow_style=False))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    path_version = args.version if args.version is not None else config['model']['version']
    ckpt_dir = os.path.join(config['paths']['checkpoint_dir'], f'v{path_version}')
    log_dir = os.path.join(config['paths']['log_dir'], f'v{path_version}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    model = build_stage1_model(config).to(device)
    criterion = V17Loss(config['loss']).to(device)
    optimizer = build_optimizer_stage1(model, config['train'], config['model'])

    # ── Dataset roots ─────────────────────────────────────────────────────────
    def _abs(root):
        return root if os.path.isabs(root) else str(ROOT_DIR / root)

    hypersim_root = _abs(config['data']['hypersim_root'])
    midintrinsic_root = _abs(config['data'].get('midintrinsic_root', '../datasets/MIDIntrinsics'))
    openrooms_root = _abs(config['data'].get('openrooms_root', '../datasets/OpenRooms'))

    split_file = config['data'].get('hypersim_split_file', 'hypersim_split.json')
    split_seed = int(config['data'].get('hypersim_split_seed', 42))
    split_ratio = float(config['data'].get('hypersim_split_ratio', 0.9))
    strict_split = bool(config['data'].get('hypersim_strict_split', True))
    hypersim_max_hdf5_retries = int(config['data'].get('hypersim_max_hdf5_retries', 1))
    hypersim_skip_corrupt_samples = bool(config['data'].get('hypersim_skip_corrupt_samples', True))
    cache_max_items = int(config['data'].get('cache_max_items', 512))
    crop_mode_train = str(config['data'].get('crop_mode_train', 'random'))
    crop_mode_val = str(config['data'].get('crop_mode_val', 'center'))

    # c·S colored-relight pairs on Hypersim — ITERATION-gated. OFF during the initial
    # warmup (clean supervised anchor), ON from `hypersim_color_pair_start_iter` onward
    # (set to ssi_warmup_iters so c·S begins the moment shading SSI is online — the
    # yellow-cast leak forms during Phase 1, so we correct it as it forms rather than
    # waiting for the Phase-2 boundary). Gated here, not via sampling weights, because
    # Hypersim has weight>0 in every phase.
    _cp_prob = float(config['data'].get('hypersim_color_pair_prob', 0.0))
    _cp_start_iter = int(config['data'].get('hypersim_color_pair_start_iter', 0))

    def _weights_key_for_step(s):
        if s < phase1_iters:
            return 'sampling_weights_phase1'
        elif s < phase2_iters:
            return 'sampling_weights_phase2'
        return 'sampling_weights_phase3'

    def _build_train_loader(weights_key, use_hyp_cp):
        hyp_cp_prob = _cp_prob if use_hyp_cp else 0.0
        return get_mixed_loader(
            data_roots={'hypersim': hypersim_root, 'midintrinsic': midintrinsic_root,
                        'openrooms': openrooms_root},
            batch_size=int(config['train']['batch_size']),
            split='train',
            num_workers=int(config['train'].get('num_workers', 4)),
            input_size=int(config['train']['input_size']),
            cache_max_items=cache_max_items,
            mix_weights=config['train'].get(weights_key, {'hypersim': 1.0}),
            split_file=split_file,
            split_seed=split_seed,
            split_ratio=split_ratio,
            strict_split=strict_split,
            max_hdf5_retries=hypersim_max_hdf5_retries,
            skip_corrupt_samples=hypersim_skip_corrupt_samples,
            use_mid_paired=bool(config['data'].get('use_mid_paired', False)),
            mid_pair_mode=str(config['data'].get('mid_pair_mode', 'raw')),
            mid_chromatic_aug=bool(config['data'].get('mid_chromatic_aug', False)),
            mid_raw_color_pair=bool(config['data'].get('mid_raw_color_pair', False)),
            hypersim_color_pair_prob=hyp_cp_prob,
            hypersim_color_tint_min=float(config['data'].get('hypersim_color_tint_min', 0.8)),
            hypersim_color_tint_max=float(config['data'].get('hypersim_color_tint_max', 1.25)),
        )

    def infinite_loader(dl):
        while True:
            for b in dl:
                yield b

    # train_loader / train_iter are built after the resume block (once start_step is
    # known) so the phase weights + the c·S gate are correct on resume too.
    val_num_workers = max(1, int(config['data'].get('val_num_workers', config['train'].get('val_num_workers', 2))))
    val_cache_max_items = max(0, int(config['data'].get('val_cache_max_items', 64)))
    val_loader = get_hypersim_loader(
        root_dir=hypersim_root,
        batch_size=int(config['train']['batch_size']),
        split='val',
        num_workers=val_num_workers,
        input_size=int(config['train']['input_size']),
        cache_max_items=val_cache_max_items,
        crop_mode_train=crop_mode_train,
        crop_mode_val=crop_mode_val,
        split_file=split_file,
        split_seed=split_seed,
        split_ratio=split_ratio,
        strict_split=strict_split,
        max_hdf5_retries=hypersim_max_hdf5_retries,
        skip_corrupt_samples=hypersim_skip_corrupt_samples,
    )

    # ── Training schedule ──────────────────────────────────────────────────────
    max_iters = int(config['train'].get('extend_iterations', 60000))
    grad_accum_steps = max(1, int(config['train'].get('grad_accum_steps', 1)))
    grad_clip_max_norm = float(config['train'].get('grad_clip_max_norm', 1.0))
    use_cosine_lr = bool(config['train'].get('use_cosine_lr', True))
    lr_eta_min = float(config['train'].get('lr_eta_min', 1.0e-7))
    val_interval_iters = int(config['train'].get('val_interval_iters', 2500))
    ckpt_interval_iters = int(config['train'].get('checkpoint_interval_iters', 1000))
    val_example_images = int(config['train'].get('val_example_images', 3))
    val_example_indices = config['train'].get('val_example_indices', [])
    max_val_batches = config['train'].get('max_val_batches', None)
    if max_val_batches is not None:
        max_val_batches = int(max_val_batches)
    compute_val_losses = bool(config['train'].get('compute_val_losses', True))
    ssi_warmup = int(config['loss'].get('ssi_warmup_iters', 3000))

    phase1_iters = int(config['train'].get('phase1_iterations', -1))
    phase2_iters = int(config['train'].get('phase2_iterations', -1))
    # Phase-end iterations whose checkpoints must NEVER be pruned.
    phase_end_iters = {phase1_iters, phase2_iters, max_iters}

    def get_phase_info(s):
        if s < phase1_iters:
            return 1, phase1_iters
        elif s < phase2_iters:
            return 2, phase2_iters
        return 3, max_iters

    # ── Resume (V20 trains from scratch by default) ────────────────────────────
    start_step = 0
    resume_path = _resolve_resume_path(args.resume, args.auto_resume, ckpt_dir)
    if resume_path:
        skip_opt = args.skip_optimizer or config['train'].get('skip_optimizer', False)
        start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device, skip_optimizer=skip_opt)
        start_step += 1

        # --start-step override: rescale the resume point when batch_size changed mid-run.
        # The checkpoint's global_step is in the OLD bs units; this puts the run back onto the
        # NEW (rescaled) schedule so phase boundaries, the cosine LR, and the dataset gates all
        # line up. Adam moments transfer cleanly because effective batch is unchanged. Example:
        # bs2 iter_10000 (20k samples / 1250 updates) → bs4 step 5000 (same 20k samples / 1250 updates).
        if args.start_step is not None:
            print(f"[resume] overriding start_step {start_step} → {args.start_step} "
                  f"(batch-size rescale; ckpt global_step is in old-bs units)")
            start_step = int(args.start_step)

        if args.reset_lr or config['train'].get('reset_lr', False):
            base_lr = float(config['train']['lr'])
            multiplier = float(config['model'].get('backbone_lr_multiplier', 1.0))
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[0]['lr'] = base_lr * multiplier
                optimizer.param_groups[1]['lr'] = base_lr
            else:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr
            print(f"Reset learning rate to {base_lr}")

    # Build the training loader now that start_step is known, so phase weights and the
    # c·S gate are correct on resume (not just from-scratch at step 0).
    train_loader = _build_train_loader(
        _weights_key_for_step(start_step), use_hyp_cp=(start_step >= _cp_start_iter)
    )
    train_iter = iter(infinite_loader(train_loader))

    # ── IIW ordinal-hinge fine-tune loader (off unless lambda_ordinal_iiw>0) ──────────────
    # Joint WHDR fine-tune: a dedicated IIW human-judgment batch per step feeds the ordinal
    # hinge while the Hypersim+MID constancy losses stay on. Train split = COMPLEMENT of the
    # eval test split (no leakage). See src/data/iiw_dataset.py + src/configs/v20_iiw_ft.yaml.
    iiw_iter = None
    if float(config['loss'].get('lambda_ordinal_iiw', 0.0)) > 0:
        from data.iiw_dataset import get_iiw_loader
        iiw_root = _abs(config['data'].get('iiw_root', 'tests/testing_data/iiw-dataset/data'))
        iiw_loader = get_iiw_loader(
            iiw_root, split='train',
            batch_size=int(config['train'].get('iiw_batch_size', 2)),
            input_size=int(config['train']['input_size']),
            num_workers=max(1, int(config['train'].get('num_workers', 4)) // 2),
        )
        iiw_iter = iter(infinite_loader(iiw_loader))
        print(f"[IIW] ordinal-hinge fine-tune ENABLED: {len(iiw_loader.dataset)} train images "
              f"(λ={config['loss']['lambda_ordinal_iiw']}), root={iiw_root}")

    scheduler = None
    if use_cosine_lr:
        total_opt_steps = max(1, math.ceil(max_iters / grad_accum_steps))
        completed_opt_steps = max(0, start_step // grad_accum_steps)
        for pg in optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_opt_steps, eta_min=lr_eta_min, last_epoch=completed_opt_steps - 1
        )

    print(f"Start step: {start_step}, max step: {max_iters}")
    print(f"Phase ends (checkpoints kept permanently): {sorted(phase_end_iters)}")

    running = {}
    current_phase, current_phase_end = get_phase_info(start_step)
    train_pbar = tqdm(desc=f'Phase {current_phase}', total=current_phase_end - start_step, dynamic_ncols=True)
    scaler = GradScaler('cuda')

    for step in range(start_step, max_iters):
        if step == current_phase_end:
            train_pbar.close()
            current_phase, current_phase_end = get_phase_info(step)
            train_pbar = tqdm(desc=f'Phase {current_phase}', total=current_phase_end - step, dynamic_ncols=True)

        if step == phase1_iters:
            print(f"--- Switching to Phase 2 Sampling Weights at step {step} ---")
            del train_iter, train_loader
            gc.collect()
            train_loader = _build_train_loader('sampling_weights_phase2',
                                               use_hyp_cp=(step >= _cp_start_iter))
            train_iter = iter(infinite_loader(train_loader))

        if step == phase2_iters:
            print(f"--- Switching to Phase 3 Sampling Weights at step {step} ---")
            del train_iter, train_loader
            gc.collect()
            train_loader = _build_train_loader('sampling_weights_phase3',
                                               use_hyp_cp=(step >= _cp_start_iter))
            train_iter = iter(infinite_loader(train_loader))

        # c·S colored-relight pairs turn ON here (iteration-gated). Rebuild once, keeping
        # the current phase's sampling weights. Skipped if it coincides with a phase
        # boundary (that rebuild already enables c·S) or if we resumed past this iteration
        # (the initial build already enabled it).
        if (step == _cp_start_iter and step > start_step
                and step not in (phase1_iters, phase2_iters) and _cp_prob > 0.0):
            print(f"--- Enabling Hypersim c·S color pairs at step {step} ---")
            del train_iter, train_loader
            gc.collect()
            train_loader = _build_train_loader(_weights_key_for_step(step), use_hyp_cp=True)
            train_iter = iter(infinite_loader(train_loader))

        batch = next(train_iter)
        iiw_batch = next(iiw_iter) if iiw_iter is not None else None
        # backward (split: main+CARI, shadow, then the IIW hinge) happens INSIDE train_one_step so the
        # forward graphs never co-reside — peak GPU memory = 2 graphs (V17+CARI level), not 3.
        losses = train_one_step(model, batch, criterion, device, step, ssi_warmup,
                                scaler=scaler, grad_accum_steps=grad_accum_steps,
                                iiw_batch=iiw_batch)

        if (step + 1) % grad_accum_steps == 0 or step == max_iters - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        train_pbar.set_postfix({'loss': f"{losses['loss_total'].item():.4f}"})
        train_pbar.update(1)

        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v.item()

        if step % int(config['train']['log_interval']) == 0:
            _log_ordered_scalars(writer, losses, step, tag_prefix=None)
            if len(optimizer.param_groups) > 1:
                writer.add_scalar('0. Training/lr_backbone', float(optimizer.param_groups[0]['lr']), step)
                writer.add_scalar('0. Training/lr_heads', float(optimizer.param_groups[1]['lr']), step)
            else:
                writer.add_scalar('0. Training/lr', float(optimizer.param_groups[0]['lr']), step)

        # ── Checkpoint + retention ─────────────────────────────────────────────
        # Force a save at every phase end (so the phase-final state always exists,
        # even if a phase boundary is not a multiple of ckpt_interval_iters).
        is_phase_end = (step + 1) in phase_end_iters
        if (step + 1) % ckpt_interval_iters == 0 or is_phase_end:
            avg = {k: running[k] / ckpt_interval_iters for k in running}
            for k in running:
                running[k] = 0.0
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_iter_{step+1}.pth')
            save_checkpoint(model, optimizer, avg, config, ckpt_path, global_step=step)
            save_checkpoint(model, optimizer, avg, config,
                            os.path.join(ckpt_dir, 'checkpoint_latest.pth'), global_step=step)

            # Keep max 2 INTRA-phase checkpoints; phase-end checkpoints are kept forever.
            all_ckpts = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.startswith('checkpoint_iter_') and f.endswith('.pth')
            ]
            prunable = [c for c in all_ckpts if _extract_iter_from_name(c) not in phase_end_iters]
            prunable.sort(key=_extract_iter_from_name)
            while len(prunable) > 2:
                oldest = prunable.pop(0)
                try:
                    os.remove(oldest)
                    print(f"Deleted old checkpoint: {oldest}")
                except Exception as e:
                    print(f"Failed to delete {oldest}: {e}")
            if is_phase_end:
                print(f"[phase-end] kept permanently: {ckpt_path}")

        # ── Validation + qualitative viz ───────────────────────────────────────
        if (step + 1) % val_interval_iters == 0:
            torch.cuda.empty_cache()
            vloss = validate(
                model, val_loader, criterion, device, step + 1, writer,
                val_example_images=val_example_images, val_example_indices=val_example_indices,
                max_val_batches=max_val_batches, compute_val_losses=compute_val_losses,
                example_root='3. Examples',
            )
            pretty = ", ".join([f"{k}={v:.4f}" for k, v in vloss.items()])
            print(f"[{step+1}] val: {pretty}")

            # MID test viz — the CARI primary eval set (watch desaturation per ckpt).
            torch.cuda.empty_cache()
            if midintrinsic_root and os.path.isdir(midintrinsic_root):
                try:
                    from src.data.midintrinsic_dataset import MIDIntrinsicDataset
                    mid_vis_ds = MIDIntrinsicDataset(
                        root_dir=midintrinsic_root, split='test',
                        input_size=int(config['train']['input_size']), use_paired=False,
                    )
                    if len(mid_vis_ds) > 0:
                        _log_dataset_examples(
                            mid_vis_ds, model, device, writer, step + 1,
                            n_samples=7, seed=42, example_root='3. Examples',
                            dataset_tag='MIDIntrinsics',
                        )
                except Exception as exc:
                    print(f"[warn] MID visualization failed at step {step+1}: {exc}")

            # OpenRooms viz — synthetic cross-render domain.
            torch.cuda.empty_cache()
            if openrooms_root and os.path.isdir(openrooms_root):
                try:
                    from src.data.openrooms_dataset import OpenRoomsDataset
                    or_vis_ds = OpenRoomsDataset(
                        root_dir=openrooms_root, split='test',
                        input_size=int(config['train']['input_size']),
                    )
                    if len(or_vis_ds) > 0:
                        _log_dataset_examples(
                            or_vis_ds, model, device, writer, step + 1,
                            n_samples=3, seed=42, example_root='3. Examples',
                            dataset_tag='OpenRooms',
                        )
                except Exception as exc:
                    print(f"[warn] OpenRooms visualization failed at step {step+1}: {exc}")

    print('Training completed')
    writer.close()


if __name__ == '__main__':
    main()
