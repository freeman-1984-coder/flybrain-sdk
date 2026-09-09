/** Design contract only. Nothing in this package executes a simulation yet. */
export type BackendName = "cpu" | "wasm" | "cuda";
export type ToySensoryChannel = "food" | "looming_left" | "looming_right" | "touch";
export interface LIFConfig {
  dtMs?: number;
  tauMs?: number;
  rest?: number;
  reset?: number;
  threshold?: number;
  refractoryMs?: number;
  inputGain?: number;
  rateTauMs?: number;
  actionRateHz?: number;
}
export interface LoadOptions {
  model?: "toy" | URL;
  backend?: BackendName;
  config?: LIFConfig;
}
export interface Stimulus {
  channel: ToySensoryChannel | (string & {});
  strength?: number;
  durationMs?: number;
}
export interface MotorAction {
  walk: number;
  turnLeft: number;
  turnRight: number;
  jump: number;
}
export interface SimulationState {
  tick: number;
  timeMs: number;
  voltage: Float64Array;
  spikes: Uint8Array;
  ratesHz: Float64Array;
}
/** Future implementation must translate camelCase API fields to the Python JSON schema. */
export interface FlyBrainInstance {
  readonly backend: BackendName;
  readonly state: SimulationState;
  stimulate(stimulus: Stimulus): this;
  clearStimuli(): void;
  step(steps?: number): SimulationState;
  action(): MotorAction;
  save(): Uint8Array;
  dispose(): void;
}
export interface FlyBrainFactory {
  load(options?: LoadOptions): Promise<FlyBrainInstance>;
  restore(checkpoint: Uint8Array, backend?: BackendName): Promise<FlyBrainInstance>;
}
