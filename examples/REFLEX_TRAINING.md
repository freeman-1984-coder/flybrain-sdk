# Reflex School: train an external action readout

Run from the repository checkout:

```sh
pip install -e '.[dev]'
python examples/train_reflex.py --backend cpu --output reflex-training.json
# On a validated CUDA host, choose --backend cuda instead.
```

Open the generated `reflex-training.html` to replay the same held-out obstacles
before and after training. It also writes portable readout weights to
`reflex-training.checkpoint.json` and full results to JSON. The HTML uses only
embedded results, needs no server/GPU, and does not perform live inference.

The pinned 313-neuron MaleCNS escape circuit receives left/right looming pulses
with randomized amplitude and weak opposite-side interference. After 80 ms,
two GF firing rates and a constant bias become the only controller features.
The controller chooses left or right. Escaping opposite the obstacle gets reward
1; moving toward it gets 0. Batched REINFORCE updates three external weights for
24 rounds using 64 training trials. A separate seed generates 80 balanced test
trials. Their results never select epochs or update weights. Zero initial weights
give a 50% deterministic tie-break baseline on this balanced task.

With CUDA selected, both LIF simulation and readout optimization execute through
CuPy on the GPU. Stimulus generation, reporting and held-out metrics run on CPU.
This tiny experiment does not imply GPU speedup. All seeds, weights, rates and
actions are saved so the result can be inspected and reproduced.

**Scientific scope:** the anatomical connectivity stays fixed. The learning is
an engineered external readout, not biological synaptic plasticity. LIF parameters,
sensory projections, reward and game rules are designed by us. Trials reset the
brain and test only a two-choice reflex; this is not a trained whole fruit fly or
evidence of complex game skill. The animation stretches each saved 80 ms response
for visibility; it is an illustration of recorded decisions, not physical replay.
