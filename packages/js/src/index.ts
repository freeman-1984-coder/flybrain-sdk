/** Float64 CPU reference implementation. No WASM, CUDA, network or UI dependency. */
export type BackendName = "cpu" | "wasm" | "cuda";
export interface Connectome {
  schema_version: 1;
  name: string;
  neuron_ids: string[];
  synapses: { pre: string; post: string; weight: number }[];
  sensory: Record<string, string[]>;
  motor: Record<string, string[]>;
  provenance?: Record<string, string>;
  annotations?: Record<string, Record<string, string>>;
}
export interface LIFConfig {
  dt_ms: number; tau_ms: number; rest: number; reset: number;
  threshold: number; refractory_ms: number; input_gain: number;
  rate_tau_ms: number; action_rate_hz: number;
}
export interface ModelBundle {
  format: "flybrain-model-bundle"; schema_version: 1;
  model: Connectome; config: LIFConfig; model_sha256: string;
}
export interface SimulationState {
  tick: number; time_ms: number;
  voltage: number[]; spikes: boolean[]; rates_hz: number[];
}
export interface Pulse { ids: string[]; amplitude: number; remaining_steps: number }
export interface Checkpoint {
  format: "flybrain-js-checkpoint"; schema_version: 1;
  dynamics_revision: "lif-exact-v1";
  model: Connectome; config: LIFConfig;
  state: SimulationState & { refractory: number[]; silenced: boolean[] };
  pending_currents: Pulse[];
  readout: { channels: Record<string, string[]>; scale_hz: number };
}
const defaults: LIFConfig = {
  dt_ms: 1, tau_ms: 10, rest: 0, reset: 0, threshold: 1,
  refractory_ms: 2, input_gain: 2, rate_tau_ms: 50, action_rate_hz: 100,
};
function finite(x: unknown, label: string): number {
  if (typeof x !== "number" || !Number.isFinite(x)) throw new Error(`${label} must be finite`);
  return x;
}
function integer(x: unknown, label: string, min = 1): number {
  const n = finite(x, label);
  if (!Number.isSafeInteger(n) || n < min) throw new Error(`${label} must be an integer >= ${min}`);
  return n;
}
function clone<T>(value: T): T { return JSON.parse(JSON.stringify(value)); }
function record(x: unknown, label: string): asserts x is Record<string, unknown> {
  if (!x || typeof x !== "object" || Array.isArray(x)) throw new Error(`${label} must be an object`);
}

export class FlyBrain {
  readonly backend = "cpu" as const;
  readonly dynamicsRevision = "lif-exact-v1" as const;
  private readonly graph: Connectome;
  private readonly cfg: LIFConfig;
  private readonly index: Map<string, number>;
  private readonly pre: Int32Array;
  private readonly post: Int32Array;
  private readonly weights: Float64Array;
  private readonly leak: number;
  private readonly rateDecay: number;
  private readonly hold: number;
  private v: Float64Array;
  private spikes: Uint8Array;
  private refractory: Float64Array;
  private rates: Float64Array;
  private silenced: Uint8Array;
  private pending: Pulse[] = [];
  private channels: Record<string, string[]>;
  private scale: number;
  private ticks = 0;

  static load(data: Connectome | ModelBundle, options: {
    backend?: BackendName; config?: Partial<LIFConfig>
  } = {}): FlyBrain {
    if (options.backend && options.backend !== "cpu") throw new Error(`${options.backend} is not implemented`);
    if ("format" in data) {
      if (data.format !== "flybrain-model-bundle" || data.schema_version !== 1) throw new Error("Unsupported model bundle");
      return new FlyBrain(data.model, { ...data.config, ...options.config });
    }
    return new FlyBrain(data, options.config);
  }

  private constructor(model: Connectome, config: Partial<LIFConfig> = {}) {
    if (model.schema_version !== 1 || typeof model.name !== "string" || !model.name) throw new Error("Invalid connectome");
    if (!Array.isArray(model.neuron_ids) || !model.neuron_ids.length ||
        model.neuron_ids.some(n => typeof n !== "string" || !n) ||
        new Set(model.neuron_ids).size !== model.neuron_ids.length) throw new Error("Neuron IDs must be unique strings");
    this.index = new Map(model.neuron_ids.map((n, i) => [n, i]));
    const cfg = { ...defaults, ...config };
    for (const name of Object.keys(config)) if (!Object.hasOwn(defaults, name)) throw new Error(`Unknown config ${name}`);
    for (const [k, v] of Object.entries(cfg)) finite(v, k);
    for (const k of ["dt_ms", "tau_ms", "input_gain", "rate_tau_ms", "action_rate_hz"] as const) {
      if (cfg[k] <= 0) throw new Error(`${k} must be positive`);
    }
    if (cfg.refractory_ms < 0 || cfg.reset >= cfg.threshold || cfg.rest >= cfg.threshold) throw new Error("Invalid LIF bounds");
    this.hold = integer(Math.ceil(cfg.refractory_ms / cfg.dt_ms), "refractory ticks", 0);
    this.cfg = cfg;
    this.leak = Math.exp(-cfg.dt_ms / cfg.tau_ms);
    this.rateDecay = Math.exp(-cfg.dt_ms / cfg.rate_tau_ms);
    if (!Array.isArray(model.synapses)) throw new Error("Missing synapses");
    for (const edge of model.synapses) {
      this.resolve([edge.pre]); this.resolve([edge.post]); finite(edge.weight, "weight");
    }
    for (const ports of [model.sensory, model.motor]) {
      record(ports, "ports");
      for (const [name, ids] of Object.entries(ports)) {
        if (!name) throw new Error("Empty port name"); this.resolve(ids);
      }
    }
    record(model.annotations ?? {}, "annotations");
    for (const [id, values] of Object.entries(model.annotations ?? {})) {
      this.resolve([id]); record(values, "annotation");
      if (Object.values(values).some(v => typeof v !== "string")) throw new Error("Annotations must be strings");
    }
    this.graph = clone(model);
    this.pre = Int32Array.from(model.synapses, e => this.index.get(e.pre)!);
    this.post = Int32Array.from(model.synapses, e => this.index.get(e.post)!);
    this.weights = Float64Array.from(model.synapses, e => e.weight);
    const n = this.index.size;
    this.v = new Float64Array(n).fill(cfg.rest);
    this.spikes = new Uint8Array(n);
    this.refractory = new Float64Array(n);
    this.rates = new Float64Array(n);
    this.silenced = new Uint8Array(n);
    this.channels = clone(model.motor);
    this.scale = cfg.action_rate_hz;
  }

  get tick(): number { return this.ticks; }
  get timeMs(): number { return this.ticks * this.cfg.dt_ms; }
  get model(): Connectome { return clone(this.graph); }
  get config(): LIFConfig { return { ...this.cfg }; }
  get state(): SimulationState {
    return { tick: this.tick, time_ms: this.timeMs, voltage: [...this.v],
      spikes: Array.from(this.spikes, Boolean), rates_hz: [...this.rates] };
  }
  private resolve(ids: string[]): number[] {
    if (!Array.isArray(ids) || !ids.length || ids.some(n => typeof n !== "string" || !this.index.has(n))) throw new Error("Expected known string neuron IDs");
    if (new Set(ids).size !== ids.length) throw new Error("Duplicate neuron IDs");
    return ids.map(n => this.index.get(n)!);
  }
  select(attributes: Record<string, string> = {}): string[] {
    const available = new Set(Object.values(this.graph.annotations ?? {}).flatMap(Object.keys));
    for (const [k, v] of Object.entries(attributes)) {
      if (!available.has(k) || typeof v !== "string") throw new Error(`Unknown annotation or invalid selector: ${k}`);
    }
    const ids = this.graph.neuron_ids.filter(n => Object.entries(attributes).every(([k, v]) => this.graph.annotations?.[n]?.[k] === v));
    if (!ids.length) throw new Error("Selector matches no neurons");
    return ids;
  }
  stimulate(input: { channel: string; strength?: number; durationMs?: number }): this {
    const { channel, strength = 1, durationMs = 20 } = input;
    if (!Object.hasOwn(this.graph.sensory, channel)) throw new Error(`Unknown sensory channel ${channel}`);
    finite(strength, "strength"); finite(durationMs, "durationMs");
    if (strength < 0 || strength > 1 || durationMs <= 0) throw new Error("Invalid stimulus bounds");
    const steps = integer(Math.ceil(durationMs / this.cfg.dt_ms), "duration steps");
    this.pending.push({ ids: [...this.graph.sensory[channel]], amplitude: strength * this.cfg.input_gain, remaining_steps: steps });
    return this;
  }
  inject(ids: string[], options: { amplitude: number; durationMs: number }): this {
    this.resolve(ids); finite(options.amplitude, "amplitude");
    const steps = finite(options.durationMs, "durationMs") / this.cfg.dt_ms;
    if (Math.abs(steps - Math.round(steps)) > 1e-9) throw new Error("Duration must align with dt");
    this.pending.push({ ids: [...ids], amplitude: options.amplitude, remaining_steps: integer(Math.round(steps), "duration steps") });
    return this;
  }
  clearStimuli(): void { this.pending = []; }
  silence(ids: string[], enabled = true): void {
    if (typeof enabled !== "boolean") throw new Error("enabled must be boolean");
    for (const i of this.resolve(ids)) this.silenced[i] = Number(enabled);
  }
  bindReadout(channels: Record<string, string[]>, scaleHz = this.cfg.action_rate_hz): this {
    record(channels, "readout"); finite(scaleHz, "scaleHz");
    if (scaleHz <= 0) throw new Error("scaleHz must be positive");
    for (const [name, ids] of Object.entries(channels)) {
      if (!name) throw new Error("Empty channel name"); this.resolve(ids);
    }
    this.channels = clone(channels); this.scale = scaleHz; return this;
  }
  action(): Record<string, number> {
    const result: Record<string, number> = Object.create(null);
    for (const [name, ids] of Object.entries(this.channels)) {
      result[name] = Math.max(0, Math.min(1, ids.reduce((s, n) => s + this.rates[this.index.get(n)!], 0) / ids.length / this.scale));
    }
    return result;
  }
  observe(ids: string[]): SimulationState & { neuron_ids: string[] } {
    const indices = this.resolve(ids);
    return { neuron_ids: [...ids], tick: this.tick, time_ms: this.timeMs,
      voltage: indices.map(i => this.v[i]), spikes: indices.map(i => !!this.spikes[i]),
      rates_hz: indices.map(i => this.rates[i]) };
  }
  advance(steps: number): void {
    integer(steps, "steps");
    if (!Number.isSafeInteger(this.tick + steps)) throw new Error("Tick overflow");
    for (let k = 0; k < steps; k++) this.oneTick();
  }
  step(steps = 1): SimulationState { this.advance(steps); return this.state; }
  private oneTick(): void {
    const n = this.index.size;
    const synaptic = new Float64Array(n), current = new Float64Array(n);
    for (let e = 0; e < this.weights.length; e++) synaptic[this.post[e]] += this.weights[e] * this.spikes[this.pre[e]];
    for (const pulse of this.pending) for (const id of pulse.ids) current[this.index.get(id)!] += pulse.amplitude;
    const v = new Float64Array(n), spikes = new Uint8Array(n), refr = new Float64Array(n), rates = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      finite(current[i], "current");
      const available = this.refractory[i] === 0 && !this.silenced[i];
      v[i] = available ? this.cfg.rest + (this.v[i] - this.cfg.rest) * this.leak + (current[i] + synaptic[i]) * (1 - this.leak) : this.cfg.reset;
      finite(v[i], "voltage");
      spikes[i] = Number(available && v[i] >= this.cfg.threshold);
      if (spikes[i]) v[i] = this.cfg.reset;
      refr[i] = spikes[i] ? this.hold : Math.max(this.refractory[i] - 1, 0);
      rates[i] = this.rates[i] * this.rateDecay + spikes[i] * (1000 / this.cfg.dt_ms) * (1 - this.rateDecay);
    }
    this.v = v; this.spikes = spikes; this.refractory = refr; this.rates = rates; this.ticks++;
    this.pending = this.pending.map(p => ({ ...p, remaining_steps: p.remaining_steps - 1 })).filter(p => p.remaining_steps > 0);
  }
  save(): Checkpoint {
    return { format: "flybrain-js-checkpoint", schema_version: 1, dynamics_revision: this.dynamicsRevision,
      model: this.model, config: this.config,
      state: { ...this.state, refractory: [...this.refractory], silenced: Array.from(this.silenced, Boolean) },
      pending_currents: clone(this.pending), readout: { channels: clone(this.channels), scale_hz: this.scale } };
  }
  static restore(data: Checkpoint): FlyBrain {
    if (data.format !== "flybrain-js-checkpoint" || data.schema_version !== 1 || data.dynamics_revision !== "lif-exact-v1") throw new Error("Unsupported JS checkpoint");
    const brain = FlyBrain.load(data.model, { config: data.config });
    const n = brain.index.size, s = data.state;
    integer(s.tick, "tick", 0);
    if (s.time_ms !== s.tick * brain.cfg.dt_ms) throw new Error("Invalid checkpoint time");
    for (const name of ["voltage", "spikes", "rates_hz", "refractory", "silenced"] as const) {
      if (!Array.isArray(s[name]) || s[name].length !== n) throw new Error(`Invalid ${name} shape`);
    }
    for (let i = 0; i < n; i++) {
      if (finite(s.voltage[i], "voltage") >= brain.cfg.threshold) throw new Error("Invalid voltage");
      const rate = finite(s.rates_hz[i], "rate");
      if (rate < 0 || rate > 1000 / brain.cfg.dt_ms) throw new Error("Invalid rate");
      if (typeof s.spikes[i] !== "boolean" || typeof s.silenced[i] !== "boolean") throw new Error("Expected boolean state");
      if (integer(s.refractory[i], "refractory", 0) > brain.hold) throw new Error("Invalid refractory counter");
    }
    if (!Array.isArray(data.pending_currents)) throw new Error("Invalid pending currents");
    for (const p of data.pending_currents) {
      brain.resolve(p.ids); finite(p.amplitude, "amplitude"); integer(p.remaining_steps, "remaining_steps");
    }
    brain.bindReadout(data.readout.channels, data.readout.scale_hz);
    brain.ticks = s.tick; brain.v.set(s.voltage); brain.spikes.set(s.spikes.map(Number));
    brain.rates.set(s.rates_hz); brain.refractory.set(s.refractory);
    brain.silenced.set(s.silenced.map(Number)); brain.pending = clone(data.pending_currents);
    return brain;
  }
}
