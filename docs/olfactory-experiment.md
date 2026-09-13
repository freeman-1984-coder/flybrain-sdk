# Full-brain food-odor experiment

**GPU probe completed; navigation not demonstrated.**
See the [negative result and parameter diagnosis](validation/flywire-odor-a16.md). The full FlyWire graph has
passed the [CUDA numerical benchmark](fullbrain-validation.md). This experiment
adds anatomically identified sensory input. It does not establish that the fly
recognizes a banana, seeks food, or flies. No CUDA is needed to build the mapping,
use the local odor adapter or run unit tests. The full-brain probe requires CUDA
and fails if unavailable; it cannot silently run a smaller circuit or use CPU.

## What “banana odor” means here

A banana is a mixture of volatile chemicals, varying with ripeness. There is no
single banana neuron. [Banana odor experiments](https://pubmed.ncbi.nlm.nih.gov/37357260/)
report ripeness and concentration dependent attraction. Our first stimulus is
**one ethyl-acetate-responsive receptor channel, Or42b → ORN_DM1**. The receptor
response has [experimental support](https://www.nature.com/articles/s41598-017-13015-w).
This is neither a reconstructed banana blend nor the complete response to ethyl
acetate, which can recruit other receptors too.

The pinned [FlyWire annotations](https://github.com/flyconnectome/flywire_annotations/tree/8587524c1748ce5ef2080822a2fc890fc03bf597)
resolve 35 left and 33 right ORN_DM1 cells. All 68 exact root IDs occur in the
official 139,255-cell v783 graph. The input population is small; **the simulated
network remains the complete graph**, including all edges used by the full-brain
benchmark. IDs are preserved as decimal strings, avoiding rounding above 2^53.
Unannotated sides, unknown selected IDs and source checksum changes fail closed.

The mapping also identifies 685 ALPNs (antennal-lobe projection neurons), 96
MBONs, 1,303 descending neurons and one DNa02 per side for observation. A group
label describes anatomical identity; it does not establish that every group
member participates in this particular odor response.

## Reproduce the mapping and GPU probe

Start with the CUDA development branch and the full-brain data/environment
instructions in [fullbrain-validation.md](fullbrain-validation.md). Download the
annotation file into your experiment folder, outside the repository:

```sh
curl --fail --location \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/Supplemental_file1_neuron_annotations.tsv \
  --output annotations.tsv
python scripts/build_olfactory_map.py \
  --annotations annotations.tsv \
  --roots data/proofread_root_ids_783.npy \
  --output results/olfactory-map.json
python scripts/probe_fullbrain_odor.py \
  --annotations annotations.tsv --data-dir data \
  --output results/fullbrain-odor.json
```

Annotation SHA-256:
`9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be`.
The output includes the pinned source URL, commit, checksum, selections and all
assumptions. The annotation repository does not specify a license in its pinned
root; **do not assume our MIT license covers upstream annotations**. We download
them at the user's request and do not bundle the annotation table in the SDK.
Cite Schlegel et al. (2024), Matsliah et al. (2024), Berg et al. (2025/2026), and
Dorkenwald et al. (2024), as requested by the source repository. The graph has
separate source/license metadata in the full-brain data instructions.

## Sensory and world assumptions

`flybrain.olfaction.OlfactoryDrive` takes two local antenna concentration samples.
It injects only the selected sensory neurons. It receives no banana coordinates,
target bearing, desired turn or reward. Its stateless response is
`current = 5 * concentration / (0.2 + concentration)` with dimensionless units;
these engineering defaults are **not fitted dose-response parameters**, pA, ppm,
or measured spike rates. A concentration of one produces current 4.1667. Other
channels can be added explicitly by creating additional adapters and summing
their vectors. There is intentionally no scientifically unqualified `banana`
preset.

`sample_odor(antenna_xyz, source_xyz)` is a game-side isotropic Gaussian field.
Sample it separately at each antenna. It models neither turbulent plumes nor
obstacle-induced airflow. The body/world owns spatial coordinates; the neural
controller receives concentrations only. The initial world will be a kinematic
body in a voxel scene, not a reconstruction of flight musculature or a full
ventral nerve cord. FlyWire v783 is a whole-brain dataset, not the whole animal's
nervous system.

## Falsifiable first gate

Five conditions reset to the identical resting checkpoint: no odor, left odor,
right odor, bilateral odor, and bilateral odor with the 68 ORNs silenced. Each
has 100 ms pre-stimulus, 500 ms stimulus, and 200 ms recovery. The graph, initial
state and dynamics stay fixed, with no background current, noise or tuning.
Spikes are counted every tick and rates/voltages sampled every 10 ms. Reports
retain every condition, the hardware, graph and script hashes, including failures.

Input/silencing checks are a software validation gate. The `completed` report
status means the experiment executed, **not** that navigation was successful.
If sensory neurons fire but ALPNs or descending neurons do not, the current LIF
preset has not demonstrated transmission to motor output. This is a result to
investigate, not a reason to inject current directly into motor neurons or to
hide a seek-food rule in the renderer.

The existing benchmark uses homogeneous dimensionless LIF neurons and incoming
absolute-weight normalization. Its numerical accuracy does not validate its
physiology. Before a biological behavior claim, compare odor propagation under
published synaptic/membrane dynamics and calibrated sensory drive. A later
external learned readout must be labeled separately from innate circuit behavior.

For context, the [Minecraft project's own architecture report](https://github.com/blendi-remade/fly-brain-minecraft/blob/main/docs/ARCHITECTURE.md)
states that its odor stimulus causes activity without a walking command, so a
handwritten body reflex implements odor taxis. Its visual appeal is useful
inspiration; the video alone is not evidence of neural food seeking.
