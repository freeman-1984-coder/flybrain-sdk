# Active development goal

Move beyond the toy SDK to a reproducible, CPU-only real-connectome experiment.

The CPU path remains required. User steering on 2026-09-10 additionally requests
real optional CUDA support, reusable demo kits and a more open architecture.
The current design proposal is [RFC 0001](rfcs/0001-open-runtime-and-demo-kits.zh-CN.md).
RFC signatures remain proposals unless listed in the current API guide.
Implemented slices now include the open Python API, real model, JS CPU circuit
lab, standalone HTML templates and verified browser-to-Python replay. CUDA,
external engine adapters, streaming audio and learning remain open.
The live browser game now shares the JS CPU core and is checked against Python feedback.
Basic composable sessions, recorded dodge/sonification kits and full feedback replay
are implemented; CUDA remains a separate draft awaiting hardware validation.

Acceptance criteria:

1. Import at least one version-pinned real MaleCNS circuit with original IDs,
   counts, licensing, source checksums and reproducible selection.
2. Publish a small runnable model bundle separately from the package; load it by
   catalog name with explicit download opt-in, caching and SHA256 validation.
3. Document model assumptions and test neural responses, edge ablation, numerical
   continuation, performance and clean installation. Distinguish anatomy from
   physiological assumptions and game mechanics.
4. Provide an interactive demonstration and usable integration examples.
5. Expand the public GitHub Pages site into an accessible, crawlable developer
   documentation site: API, model cards, tutorials, bilingual entry points,
   sitemap, machine-readable catalog, llms.txt and plain-text documentation.
6. Validate and publicly release the code, model, website and contribution paths.
7. Separate connectome assets, dynamics, execution backends and replaceable
   input/readout/environment components. Validate these boundaries with circuit,
   interactive-game and audio use cases sharing the same core.
8. Provide runnable, exportable demo templates with pinned recipes, inspection,
   replay and a small CPU path. Clearly label planned templates until shipped.
9. Implement optional CUDA and validate it on actual NVIDIA hardware before
   advertising support. A short rented-GPU run is an acceptable route; its
   provider, price, spending cap and shutdown must be resolved before provisioning.

Search discoverability is part of the goal. Technical accessibility, accurate
content and machine-readable documentation can be delivered and verified; search
indexing and rankings remain decisions of external search providers.
