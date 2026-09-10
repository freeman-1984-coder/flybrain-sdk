# flybrain-sdk JavaScript CPU runtime

**No CUDA required.** Float64 LIF simulation for Node and browsers, with the same
model JSON and dynamics as the Python reference. Try the
[circuit lab](https://freeman-1984-coder.github.io/flybrain-sdk/lab.html) or download
an editable, standalone HTML demo there.

This alpha package has `private: true` and is **not published on NPM**. CPU works;
`wasm` and `cuda` are reserved backend names and fail explicitly.

From a repository checkout:

```sh
npm ci
npm test
```

```js
import { FlyBrain } from './dist/index.js';
// Caller supplies parsed, trusted model JSON. load() does not fetch anything.
const brain = FlyBrain.load(bundle, { backend: 'cpu' });
brain.stimulate({ channel: 'looming_left', durationMs: 100 });
brain.advance(100); // integer ticks; default dt is 1 ms
console.log(brain.action());
const restored = FlyBrain.restore(brain.save());

const gf = brain.select({ cell_type: 'DNp01' });
brain.bindReadout({ flash: gf }, 100);
brain.inject(gf, { amplitude: -2, durationMs: 20 });
brain.silence(gf, true);
console.log(brain.observe(gf));
```

`step(ticks=1)` returns a full observation. `advance(ticks)` avoids returning arrays.
`action()` reads only configured output channels. `clearStimuli()` clears *all*
pending named and direct current pulses; Python exposes separate sensory and
direct-current clearing. IDs are strings, including IDs beyond JS safe integers.
The core does not authenticate model hashes; the website loader verifies pinned
asset bytes before calling it. Load trusted data in other applications.

JS checkpoints are self-contained `flybrain-js-checkpoint` objects and are **not**
Python checkpoint files. The lab exports a command recording for Python replay.
See [the browser guide](../../docs/browser-lab.md) for exact semantics, parity
coverage, build instructions, licensing and limits. Run the core in a worker
when connecting a continuously updating game UI.
