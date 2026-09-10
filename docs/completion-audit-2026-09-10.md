# Delivery audit — 2026-09-10

This is an evidence inventory, not a declaration that every requested capability
is complete. The published CPU baseline is
[v0.4.0a3](https://github.com/freeman-1984-coder/flybrain-sdk/releases/tag/v0.4.0a3),
commit `3aaa0c0d027dc3d2405be276c0e4a32447a4d189`.
Godot and CUDA development remain separately reviewable in PRs.

| Requirement | Evidence inspected | Current conclusion |
| --- | --- | --- |
| CPU install and unified brain API | Published wheel, clean installed 0.4.0a3 environment; `FlyBrain.load`, stimulation, stepping, action and file save/restore exercised; source layout, package metadata and MIT license | Delivered for the documented LIF scope. CUDA is not needed. |
| Real data importer, lineage and model | `models/male-cns-escape-v1/recipe.json`, model card and original-ID/contact-count tests; importer aggregation, explicit signs and checksum rejection tests | Delivered: 313 cells, 20,607 edges, 79,112 contacts. Anatomical data is separate from assumed dynamics and engineered control. The 1.1 GB full-source rebuild was not repeated during this audit. |
| On-demand catalog and integrity | Fresh temporary cache with the clean release installation: loading without download permission rejected; explicit download succeeded and produced fingerprint `47a0c91b3eba9c606c29faef58fe15c8846fc049c4314a5e3b6912ab22c5e68a`; registry corruption/cache tests | Delivered for the ready real subgraph. Raw-data catalog entries are not advertised as runnable models. |
| Numerical continuation and within-model checks | Clean release resumed after tick 35 and matched all 65 following ticks; real-model pulse/decay, GF-edge ablation and checkpoint tests | Software behavior verified within the declared model; no animal-data or learning validation claimed. |
| CPU performance | Model-card benchmark: five 1,000-tick runs, recorded environment and excluded loading cost; benchmark script available | A scoped small-circuit measurement is published. No full-brain, CUDA or universal real-time claim. |
| Shared core and open application interfaces | Python selected-cell drive/observation/readout and Session components; JS runtime plus per-tick Python comparisons; dodge and tone environments | Delivered for LIF CPU and replaceable numeric input/readout/environment components. Additional dynamics and learned readouts remain proposals. |
| Runnable, editable and exportable demos | Installed project generator, recipe-edit tests, verified feedback replay, tone WAV, browser circuit lab and live game | Delivered CPU examples. Browser and Python checkpoint formats are distinguished. Browser automatic file-download completion remains unconfirmed in controlled browsers; copyable JSON export and file imports were verified. |
| Public repository, release, website and contributions | GitHub release asset digests, Pages configuration/workflow, static documentation, model cards, contribution guide and issue/PR templates | Published. PyPI/NPM publication is not claimed. Search Console verification, indexing and search ranking are not established by these artifacts. |
| Engine adapter | [PR #5](https://github.com/freeman-1984-coder/flybrain-sdk/pull/5): actual Godot processes, toy/real 200-frame parity, separate-process restore and fault tests | Development preview. Native window layout and interactive file-dialog/save/pause checks remain a release gate. |
| Optional CUDA | [PR #4](https://github.com/freeman-1984-coder/flybrain-sdk/pull/4): separate implementation and prepared device-validation workflow | Incomplete. CPU checks do not prove GPU kernel compilation, numerical parity or performance. Actual NVIDIA execution is required. |

The release wheel digest inspected in GitHub is
`98957bc99bad2460b31c4f86d757ddffca9348652c2430b176fdc0005654c9ad`;
the source archive digest is
`aedb453988cc6dec928bc301b1d9c254abea142991c77f9eae416fedb3d6f80f`.
The fresh-cache continuation check used that release installation, rather than the
unreleased adapter code in the working checkout.

## Remaining closure work

1. Finish the native Godot interaction checklist in its README, repair any failures,
   then merge and release the adapter with updated website installation guidance.
2. Resolve an available NVIDIA machine or a rental account/provider and explicit
   spending cap. Run the prepared real-device gate, retain its device/software and
   numerical evidence, shut down rented resources, and merge CUDA only if it passes.
3. Run release packaging/clean-install checks for the integrated version and verify
   the public documentation reflects the capabilities actually shipped.

Audio input, learned readouts, additional dynamics, button/pose adapters and more
models are future product directions from the design investigation. They are not
claimed as implemented, and adding them does not replace the unresolved delivery
gates above. No paid GPU resource has been started by this audit.

## 2026-09-11 Godot follow-up

The native interaction gate above is now passed on macOS with Godot 4.7.2:
run/pause held frame 62; save/restore returned to the same scene and continued;
saving while running paused at frame 114 with brain tick 2280. The native file
picker restored that checkpoint; cancel returned without changing the world.
Stopping the bridge during play stopped the scene at frame 212, zeroed applied
control and disabled Run. Restarting the bridge and scene reset to frame 0.
The native dialog also avoids the misleading permission message observed in
Godot's custom file picker. Dialog nodes are freed after selection or cancel.

This follow-up prepares v0.4.0a4; the earlier wheel digests and clean-cache audit
above remain historical evidence for v0.4.0a3. The new release includes the Godot
adapter, reusable ExternalController and updated public demo/AI-tool guidance.
CUDA still requires actual NVIDIA execution; no GPU resource has been rented.
