import { DodgeSession } from './dodge-runtime.js';
let session;
self.onmessage = ({data}) => {
  const {id,op} = data;
  try {
    let frame=null, extra={};
    if(op==='load') session=new DodgeSession(data.bundle,data.seed??7);
    else if(op==='restore') session=DodgeSession.restore(data.checkpoint);
    else if(op==='replay') { session=DodgeSession.replay(data.recording); extra.verified=true; }
    else {
      if(!session)throw new Error('Load a model first.');
      if(op==='step')frame=session.step();
      else if(op==='command')session.command(data.command);
      else if(op==='export')extra.recording=session.recording();
      else if(op==='save')extra.checkpoint=session.save();
      else throw new Error('Unknown operation.');
    }
    self.postMessage({id,ok:true,frame,environment:session.environment,
      observation:session.observe(),settings:session.settings,recordedFrames:session.recordedFrames,
      outputIds:session.outputIds,rates_hz:session.outputRates,
      modelName:session.modelName,neuronCount:session.neuronCount,...extra});
  }catch(error){self.postMessage({id,ok:false,error:error.message});}
};
