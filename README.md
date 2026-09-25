# Crazyflie KAE training

This repository trains each frozen 16-mode KAE. Its local `kae_moe` package
contains the PyTorch controller and tensor-only checkpoint tools. `Crazyflie_RL`
and `qualisys_drone_sdk` each keep their own copy of the runtime. Each repository
runs independently; only datasets and checkpoint files pass between them.
Simulator and hardware training remain in their respective repositories.

For notebook training, install this repository's dependencies from its root:

```bash
python -m pip install -e '.[notebook]'
```

This adds PyTorch, NumPy, SciPy, Matplotlib, tqdm, and Jupyter. The notebook
explicitly imports the local `kae_moe` directory, so it does not depend on an
editable installation from another checkout. Do not install this repository
into Isaac or SDK environments; use their own setup instructions.

## Train a KAE from an exported dataset

1. Open `KAE/main_walk_2.ipynb` with `KAE/` as the working directory.
2. Edit its configuration cell: `dataset_dir`, `output_path`, `device`, `seed`,
   `num_epochs`, and `batch_size`. Keep `observable_dim = 16`.
3. Run the notebook. It retains the existing one-step Koopman reconstruction
   and action losses, fits `K`, and exports the selected model after conversion
   parity checks. The learning-rate schedule now updates Adam's parameter groups.

Use a new output filename for each stage. The output includes encoder/decoder
weights, `K`, normalization, the captured spectral basis, and dataset metadata.
Do not substitute a bare external-model `state_dict`: the notebook model does
not register `K`, so that dictionary is incomplete.

| Dataset | Training observations | Action labels | Runtime KAE normalization |
| --- | --- | --- | --- |
| Isaac square teacher export | Already teacher-normalized | Teacher mean actions | Saved frozen teacher normalizer |
| Real SDK export | Raw 15D observations | Full action for an initial KAE, or correction for a later KAE | Identity |

Isaac datasets provide `metadata.json`, `state_preprocessor.pt`, `obs_log.pkl`,
and `act_log.pkl`. SDK datasets provide `dataset_metadata.json`, the same pickle
files, and optionally `kae_dataset.pt`. The loader prefers the SDK tensor file
and uses **`kae_training_actions`**, not its separately stored full policy actions.
Pickle files must be trusted local exports.

The loader validates aligned, finite 15D observations and 3D labels without
clipping labels. Correction labels can exceed `[-1, 1]`. It applies no additional
normalization to the training data. The runtime observation normalizer is attached
only during portable export: the training encoder also processes action targets
while fitting `K`, so inserting an observation transform there would corrupt
those target encodings.

Parity is checked on stored training-space observations before attaching the
normalizer, and separately on synthetic raw observations with explicit
preprocessing. Clipped teacher inputs are never treated as invertible raw data.
These checks validate conversion, not held-out controller quality.

## Manual experiment chain

```text
Crazyflie_RL: train square PPO teacher and export observation/action pairs
Koopman_decompose_ext: train square KAE (16 modes)
Crazyflie_RL: freeze KAE, train complete square MoE, export trained policy
qualisys_drone_sdk: load policy, adapt to real circle, export correction data
Koopman_decompose_ext: train circle-correction KAE (16 additional modes)
qualisys_drone_sdk: prepare expanded 32-mode MoE from previous real data
qualisys_drone_sdk: adapt to moving circle, export correction data
Koopman_decompose_ext: train moving-circle-correction KAE (16 additional modes)
qualisys_drone_sdk: prepare 48-mode MoE and adapt to random-height figure eight
```

The new correction KAE learns `final_policy_action - additive_full_KAE_stack`.
The expanded MoE is separately fitted to the **previous full policy actions**
before its next real task. After source preparation, all new training data comes
from real experiments. No experiment manifest or automatic stage runner is used.

## Checkpoints and tests

`kae_moe.checkpoints` exports `save_kae_checkpoint` / `load_kae_checkpoint` and
`save_moe_checkpoint` / `load_moe_checkpoint`. Full MoE files preserve ordered
branches, spectral bases, normalization, learned head weights, and optional
critic/value-normalizer state. They do not contain optimizer or rollout state;
native trainer checkpoints remain responsible for same-run resume.

```bash
python -m unittest discover -s tests
```
