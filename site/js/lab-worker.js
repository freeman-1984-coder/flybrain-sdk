import { FlyBrain } from './runtime.js';
let brain, bundle, modelId, assetHash, commands = [];
self.onmessage = ({data}) => {
  const {id, op} = data;
  try {
    let extra = {};
    if (op === 'load') {
      bundle = data.bundle; modelId = data.modelId; assetHash = data.assetHash;
      brain = FlyBrain.load(bundle); commands = [];
    } else {
      if (!brain) throw new Error('Load a model first.');
      if (op === 'advance') brain.advance(data.ticks);
      else if (op === 'reset') { brain = FlyBrain.load(bundle); commands = []; }
      else if (op === 'stimulate') {
        if (commands.length >= 10000) throw new Error('Recording limit reached. Export and reset to continue.');
        brain.stimulate({channel:data.channel,strength:data.strength,durationMs:data.durationMs});
        commands.push({tick:brain.tick,op:'stimulate',channel:data.channel,strength:data.strength,duration_ms:data.durationMs});
      } else if (op === 'silence') {
        if (commands.length >= 10000) throw new Error('Recording limit reached. Export and reset to continue.');
        brain.silence(data.ids,data.enabled);
        commands.push({tick:brain.tick,op:'silence',ids:data.ids,enabled:data.enabled});
      } else if (op === 'export') {
        extra.experiment = {format:'flybrain-experiment',schema_version:1,model_id:modelId,
          model_sha256:bundle.model_sha256,model_asset_sha256:assetHash,
          dynamics_revision:brain.dynamicsRevision,config:brain.config,duration_ticks:brain.tick,
          commands:structuredClone(commands),expected:{state:brain.state,action:brain.action()}};
      } else throw new Error('Unknown lab operation.');
    }
    const state = brain.state;
    self.postMessage({id,ok:true,state,action:brain.action(),recordedCommands:commands.length,...extra});
  } catch (error) { self.postMessage({id,ok:false,error:error.message}); }
};
