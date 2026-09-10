/** Reference game adapter for the same fixed-20ms Python DodgeArena recipe. */
import { FlyBrain, type Checkpoint, type ModelBundle } from "./index.js";

export interface Arena {
  type: "dodge-arena-v1"; seed: number; rng: number; frame: number; x: number;
  blocks: {x: number; y: number}[]; collisions: number; passed: number;
}
export type Command = {op: "add"; x: number; y: number} | {op: "clear"} |
  {op: "configure"; input_gain: number; output_gain: number} | {op: "silence"; enabled: boolean};
export interface DodgeCheckpoint {
  format: "flybrain-dodge-checkpoint"; schema_version: 1;
  model_sha256: string; brain: Checkpoint; environment: Arena;
  input_gain: number; output_gain: number; silenced: boolean;
}
export interface Frame {
  index: number; elapsed_ms: number; brain_tick: number;
  observation: Record<string, number>; rates_hz: number[];
  requested: {steer: number}; applied: {steer: number}; environment: Arena;
}
export interface Recording {
  format: "flybrain-dodge-recording"; schema_version: 1;
  initial: DodgeCheckpoint; events: {frame: number; command: Command}[];
  frames: Frame[]; final: DodgeCheckpoint;
}
const copy = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const clamp = (x: number, a: number, b: number) => Math.max(a, Math.min(b, x));
function bounded(x: number, a: number, b: number): number {
  if (typeof x !== "number" || !Number.isFinite(x) || x < a || x > b) throw new Error(`Expected number in ${a}..${b}`);
  return x;
}
function integer(x: number, max = Number.MAX_SAFE_INTEGER): number {
  bounded(x, 0, max); if (!Number.isSafeInteger(x)) throw new Error("Expected integer"); return x;
}
function arena(data: Arena): Arena {
  if (data.type !== "dodge-arena-v1") throw new Error("Unsupported arena");
  integer(data.seed, 2**32-1); integer(data.rng, 2**32-1); integer(data.frame);
  bounded(data.x, 0, 1); integer(data.collisions); integer(data.passed);
  if (!Array.isArray(data.blocks) || data.blocks.length > 64) throw new Error("At most 64 obstacles");
  data.blocks.forEach(b => { bounded(b.x, 0, 1); bounded(b.y, 0, 1); });
  return copy(data);
}

export class DodgeSession {
  readonly modelName: string;
  readonly neuronCount: number;
  private brain: FlyBrain;
  private world: Arena;
  private readonly inputs: string[][];
  private readonly outputs: string[];
  private readonly modelHash: string;
  private gain = 2;
  private outputGain = 1;
  private muted = false;
  private initial: DodgeCheckpoint;
  private events: Recording["events"] = [];
  private frames: Frame[] = [];

  constructor(bundle: ModelBundle, seed = 7) {
    integer(seed, 2**32-1);
    if (bundle.model.neuron_ids.length > 1000 || bundle.model.synapses.length > 100000) throw new Error("Live demo supports small circuits only");
    this.modelName = bundle.model.name; this.neuronCount = bundle.model.neuron_ids.length;
    this.brain = FlyBrain.load(bundle);
    // This adapter uses the public demo's exact period; other periods use Python Session.
    if (this.brain.config.dt_ms !== 1) throw new Error("Dodge adapter requires dt_ms=1");
    this.modelHash = bundle.model_sha256;
    if (bundle.model.name === "toy-v1") this.outputs = ["motor.turn_right", "motor.turn_left"];
    else if (this.modelHash === "47a0c91b3eba9c606c29faef58fe15c8846fc049c4314a5e3b6912ab22c5e68a") this.outputs = ["10010", "10001"];
    else throw new Error("Dodge preset requires toy-v1 or the pinned MaleCNS escape circuit");
    this.inputs = [bundle.model.sensory.looming_left, bundle.model.sensory.looming_right].map(copy);
    this.brain.observe([...new Set([...this.inputs.flat(), ...this.outputs])]);
    this.world = {type:"dodge-arena-v1",seed,rng:seed,frame:0,x:0.5,blocks:[],collisions:0,passed:0};
    this.spawn(); this.initial = this.save();
  }
  private spawn(): void {
    this.world.rng = (1664525 * this.world.rng + 1013904223) % 2**32;
    this.world.blocks.push({x:0.15 + 0.7 * this.world.rng / 2**32,y:0});
  }
  get frameIndex(): number { return this.world.frame; }
  get environment(): Arena { return copy(this.world); }
  get settings() { return {input_gain:this.gain,output_gain:this.outputGain,silenced:this.muted}; }
  get neuralState() { return this.brain.state; }
  get outputIds(): string[] { return [...this.outputs]; }
  get outputRates(): number[] { return this.brain.observe(this.outputs).rates_hz; }
  get recordedFrames(): number { return this.frames.length; }
  observe(): Record<string, number> {
    let left = 0, right = 0;
    for (const block of this.world.blocks) {
      const dx = block.x - this.world.x;
      const intensity = clamp((block.y - 0.1) / 0.65, 0, 1) * Math.max(0, 1 - Math.abs(dx)/0.6);
      if (dx <= 0) left = Math.max(left, intensity);
      if (dx >= 0) right = Math.max(right, intensity);
    }
    return {danger_left:left,danger_right:right,player_x:this.world.x,collisions:this.world.collisions,passed:this.world.passed};
  }
  command(command: Command): void {
    if (this.events.length >= 10000) throw new Error("Event limit reached. Export and reset.");
    if (command.op === "add") {
      bounded(command.x,0,1); bounded(command.y,0,0.95);
      if (this.world.blocks.length >= 64) throw new Error("At most 64 obstacles");
      this.world.blocks.push({x:command.x,y:command.y});
    } else if (command.op === "clear") this.world.blocks = [];
    else if (command.op === "configure") {
      bounded(command.input_gain,0,4); bounded(command.output_gain,0,2);
      this.gain = command.input_gain; this.outputGain = command.output_gain;
    } else if (command.op === "silence") {
      if (typeof command.enabled !== "boolean") throw new Error("Silence requires boolean");
      this.brain.silence(this.outputs, command.enabled); this.muted = command.enabled;
    } else throw new Error("Unknown command");
    this.events.push({frame:this.frameIndex,command:copy(command)});
  }
  step(): Frame {
    if (this.frames.length >= 5000) throw new Error("100-second recording limit reached. Export and reset.");
    if (this.brain.tick !== this.frameIndex * 20) throw new Error("Neural clock mismatch");
    const observation = this.observe();
    [observation.danger_left, observation.danger_right].forEach((value,i) => {
      if (value * this.gain) this.brain.inject(this.inputs[i],{amplitude:value*this.gain,durationMs:20});
    });
    this.brain.advance(20);
    const rates = this.brain.observe(this.outputs).rates_hz;
    const steer = clamp((rates[0] - rates[1]) * this.outputGain / 100,-1,1);
    const previous = this.world.x;
    this.world.x = clamp(previous + steer * 0.02,0,1);
    const remaining: Arena["blocks"] = [];
    for (const block of this.world.blocks) {
      const y = block.y + 0.8 * 0.02;
      if (y >= 1) {
        if (Math.abs(block.x - this.world.x) < 0.13) this.world.collisions++;
        else this.world.passed++;
      } else remaining.push({x:block.x,y});
    }
    this.world.blocks = remaining; this.world.frame++;
    if (this.frameIndex % 45 === 0 && this.world.blocks.length < 64) this.spawn();
    const frame: Frame = {index:this.frameIndex,elapsed_ms:this.frameIndex*20,brain_tick:this.brain.tick,
      observation,rates_hz:rates,requested:{steer},applied:{steer:(this.world.x-previous)/0.02},environment:this.environment};
    this.frames.push(copy(frame)); return frame;
  }
  save(): DodgeCheckpoint {
    return {format:"flybrain-dodge-checkpoint",schema_version:1,model_sha256:this.modelHash,
      brain:this.brain.save(),environment:this.environment,...this.settings};
  }
  recording(): Recording {
    return {format:"flybrain-dodge-recording",schema_version:1,initial:copy(this.initial),
      events:copy(this.events),frames:copy(this.frames),final:this.save()};
  }
  static restore(data: DodgeCheckpoint): DodgeSession {
    if (data.format !== "flybrain-dodge-checkpoint" || data.schema_version !== 1) throw new Error("Unsupported dodge checkpoint");
    const world = arena(data.environment);
    bounded(data.input_gain,0,4); bounded(data.output_gain,0,2);
    if (typeof data.silenced !== "boolean") throw new Error("Invalid silence state");
    const brain = FlyBrain.restore(data.brain);
    if (brain.tick !== world.frame * 20) throw new Error("Checkpoint clocks differ");
    const result = new DodgeSession({format:"flybrain-model-bundle",schema_version:1,
      model:brain.model,config:brain.config,model_sha256:data.model_sha256},world.seed);
    const outputIndices = result.outputs.map(id=>brain.model.neuron_ids.indexOf(id));
    if (outputIndices.some(i=>data.brain.state.silenced[i] !== data.silenced)) throw new Error("Silence state differs");
    result.brain = brain; result.world = world; result.gain = data.input_gain;
    result.outputGain = data.output_gain; result.muted = data.silenced;
    result.initial = result.save(); return result;
  }
  static replay(data: Recording): DodgeSession {
    if (data.format !== "flybrain-dodge-recording" || data.schema_version !== 1 ||
        !Array.isArray(data.frames) || data.frames.length > 5000 ||
        !Array.isArray(data.events) || data.events.length > 10000) throw new Error("Invalid recording");
    const result = DodgeSession.restore(data.initial);
    let cursor = 0;
    const events = () => {
      while (cursor < data.events.length && data.events[cursor].frame === result.frameIndex) {
        result.command(data.events[cursor++].command);
      }
    };
    for (const frame of data.frames) {
      events(); const actual = result.step();
      if(actual.index !== frame.index || actual.brain_tick !== frame.brain_tick || actual.elapsed_ms !== frame.elapsed_ms) throw new Error("Recording clock mismatch");
      compare(actual,frame);
    }
    events();
    if (cursor !== data.events.length) throw new Error("Unordered or out-of-range events");
    if(result.frameIndex !== data.final.environment.frame || result.brain.tick !== data.final.brain.state.tick) throw new Error("Final clock mismatch");
    compare(result.save(),data.final); return result;
  }
}

function compare(a: unknown, b: unknown): void {
  if (typeof a === "number" && typeof b === "number" && Number.isFinite(b) && Math.abs(a-b) <= 1e-9) return;
  if (a === b) return;
  if (a && b && typeof a === "object" && typeof b === "object" && Array.isArray(a) === Array.isArray(b)) {
    const aa = a as Record<string,unknown>, bb = b as Record<string,unknown>;
    const keys = Object.keys(aa);
    if (keys.length === Object.keys(bb).length && keys.every(k=>Object.hasOwn(bb,k))) {
      keys.forEach(k=>compare(aa[k],bb[k])); return;
    }
  }
  throw new Error("Recording replay mismatch");
}
