import { bytes } from './model-loader.js';
const $=id=>document.getElementById(id);
const worker=new Worker(new URL('./dodge-worker.js',import.meta.url),{type:'module'});
let sequence=0,pending=new Map(),state=null,lastFrame=null,bundle=null,running=false,epoch=0,timer=null,editing=false;
let importedModel=false;
const controls=()=>document.querySelectorAll('[data-ready],#toy,#real');
function enable(value){controls().forEach(el=>el.disabled=!value);}
function status(message,error=false){$('status').textContent=message;$('status').dataset.error=String(error);}
function call(op,data={}){return new Promise((resolve,reject)=>{
  const id=++sequence;pending.set(id,{resolve,reject});worker.postMessage({id,op,...data});
});}
worker.onmessage=({data})=>{
  const request=pending.get(data.id);if(!request)return;pending.delete(data.id);
  if(!data.ok){request.reject(new Error(data.error));return;}
  state=data;if(data.frame)lastFrame=data.frame;draw();request.resolve(data);
};
worker.onerror=()=>{pause();for(const p of pending.values())p.reject(new Error('Simulation worker failed. Reload the page.'));pending.clear();enable(false);status('Simulation worker failed. Reload the page.',true);};
function pause(){running=false;epoch++;clearTimeout(timer);$('run').textContent='Run';$('speed').textContent='Paused. Neural time does not advance.';}
function start(){if(!state||editing)return;running=true;$('run').textContent='Pause';const version=++epoch,start=performance.now(),first=state.environment.frame;
  async function tick(){
    if(!running||version!==epoch)return;const began=performance.now();
    try{await call('step');if(version!==epoch)return;
      const factor=(state.environment.frame-first)*20/(performance.now()-start);
      $('speed').textContent=`${factor.toFixed(2)}× simulated / wall time · no neural steps skipped`;
      timer=setTimeout(tick,Math.max(0,20-(performance.now()-began)));
    }catch(error){pause();status(error.message,true);}
  }tick();
}
async function edit(action,{resume=true}={}){
  if(editing)return;const wasRunning=running;pause();editing=true;enable(false);
  try{await action();}catch(error){status(error.message,true);resume=false;}
  finally{editing=false;enable(!!state);$('toy').disabled=false;$('real').disabled=false;if(wasRunning&&resume)start();}
}
async function load(model){
  await edit(async()=>{
    status(model==='toy'?'Loading the toy circuit…':'Downloading and verifying 3.8 MB…');
    let url=new URL('../toy-bundle.json',import.meta.url).href,size=null,hash=null;
    if(model!=='toy'){
      const response=await fetch(new URL('../model-catalog.json',import.meta.url));
      if(!response.ok)throw new Error('Cannot read model catalog.');
      const entry=(await response.json()).find(e=>e.id==='male-cns-escape-v1');
      const asset=entry?.assets?.model;
      if(entry?.status!=='ready'||!asset)throw new Error('Real model unavailable.');
      url=asset.url;size=asset.size_bytes;hash=asset.checksum.replace('sha256:','');
      if(!/^https:\/\//.test(url)||!/^[a-f0-9]{64}$/.test(hash))throw new Error('Invalid model integrity metadata.');
    }
    const next=JSON.parse(new TextDecoder().decode(await bytes(url,size,hash)));
    lastFrame=null;await call('load',{bundle:next});bundle=next;importedModel=false;draw();
    status('Ready. Run or step, then add obstacles to change the input.');
  },{resume:false});
}
let downloadUrl=null;
function download(name,data){
  if(downloadUrl)URL.revokeObjectURL(downloadUrl);
  const json=JSON.stringify(data);
  downloadUrl=URL.createObjectURL(new Blob([json],{type:'application/json'}));
  $('export-json').value=json;$('export-fallback').hidden=false;
  const link=$('download-file');link.href=downloadUrl;link.download=name;
  link.textContent=`Download ${name}`;$('download-note').hidden=false;link.click();
}

function draw(){
  if(!state)return;const env=state.environment,settings=state.settings,obs=state.observation;
  $('model-name').textContent=importedModel?'Imported experiment · self-described model':state.neuronCount===12?'Toy · artificial circuit':'MaleCNS · real anatomical subgraph';
  $('model-detail').textContent=`${state.neuronCount} neurons · assumed LIF dynamics · local CPU simulation`;
  $('clock').textContent=`${env.frame*20} ms · frame ${env.frame}`;
  $('score').textContent=`${env.passed} passed · ${env.collisions} collisions`;
  $('input').textContent=`Left ${obs.danger_left.toFixed(3)} · Right ${obs.danger_right.toFixed(3)}`;
  $('rates').textContent=state.rates_hz.map(n=>`${n.toFixed(1)} Hz`).join(' · ');
  $('cells').textContent=state.outputIds.join(' / ');
  $('requested').textContent=lastFrame?lastFrame.requested.steer.toFixed(3):'—';
  $('applied').textContent=lastFrame?lastFrame.applied.steer.toFixed(3):'—';
  for(const key of ['input','output']){$(`${key}-gain`).value=settings[`${key}_gain`];$(`${key}-value`).textContent=settings[`${key}_gain`].toFixed(1);}
  $('silence').setAttribute('aria-pressed',String(settings.silenced));
  $('silence').textContent=settings.silenced?'Release output cells':'Silence output cells';
  const canvas=$('arena'),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height;
  const margin=30,width=w-2*margin,top=30,bottom=h-58,height=bottom-top;
  ctx.fillStyle='#f4f7fd';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#dce5f3';ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const x=margin+width*i/5;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom+15);ctx.stroke();}
  ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(margin,bottom);ctx.lineTo(w-margin,bottom);ctx.stroke();ctx.setLineDash([]);
  for(const block of env.blocks){const x=margin+block.x*width,y=top+block.y*height;
    ctx.fillStyle='#e79b69';ctx.beginPath();ctx.roundRect(x-13,y-9,26,18,5);ctx.fill();
  }
  const x=margin+env.x*width,y=bottom;
  ctx.fillStyle=settings.silenced?'#91a0b8':'#4164df';ctx.beginPath();ctx.arc(x,y,14,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x-4,y-3,2,0,Math.PI*2);ctx.arc(x+4,y-3,2,0,Math.PI*2);ctx.fill();
  const steer=lastFrame?.applied.steer??0;
  if(Math.abs(steer)>.01){ctx.strokeStyle='#4164df';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,bottom+24);ctx.lineTo(x+steer*45,bottom+24);ctx.stroke();}
  ctx.fillStyle='#657692';ctx.font='12px system-ui';ctx.textAlign='left';ctx.fillText('Obstacles descend',margin,18);
  ctx.textAlign='right';ctx.fillText(`Player x = ${env.x.toFixed(3)}`,w-margin,h-12);
}
$('run').onclick=()=>running?pause():start();
$('step').onclick=()=>edit(()=>call('step'),{resume:false});
$('toy').onclick=()=>load('toy');$('real').onclick=()=>load('real');
$('reset').onclick=()=>edit(async()=>{lastFrame=null;await call('load',{bundle,seed:7});status('Reset to seed 7, default gains, resting brain.');},{resume:false});
$('clear').onclick=()=>edit(async()=>{await call('command',{command:{op:'clear'}});status('Obstacles cleared. Automatic spawning continues every 45 frames.');});
function add(x,y){return edit(async()=>{await call('command',{command:{op:'add',x,y}});status('Obstacle added to the live environment. Its position affects the next input.');});}
$('add').onclick=()=>add(Number($('position').value),0.7);
$('arena').onclick=event=>{
  if(!state||editing)return;const r=$('arena').getBoundingClientRect();
  add(Math.max(0,Math.min(1,(event.clientX-r.left-30)/(r.width-60))),Math.max(0,Math.min(.95,(event.clientY-r.top-30)/(r.height-88))));
};
for(const name of ['input','output'])$(`${name}-gain`).onchange=()=>{
  const input_gain=Number($('input-gain').value),output_gain=Number($('output-gain').value);
  edit(async()=>{await call('command',{command:{op:'configure',input_gain,output_gain}});status('Mapping updated. New gains apply on the next neural step.');});
};
$('silence').onclick=()=>edit(async()=>{
  const enabled=!state.settings.silenced;await call('command',{command:{op:'silence',enabled}});
  status(enabled?'Output cells silenced. Existing filtered activity decays; step or run to observe.':'Output cells released.');
});
$('save').onclick=()=>edit(async()=>{const result=await call('save');download('flybrain-dodge.checkpoint.json',result.checkpoint);status('Checkpoint prepared. Use the download link below if your browser does not save automatically.');});
$('export').onclick=()=>edit(async()=>{const result=await call('export');download('flybrain-dodge.recording.json',result.recording);status('Recording prepared. Download the JSON below, then import it to verify replay.');});
$('restore').onchange=()=>{
  const file=$('restore').files[0];if(!file)return;
  edit(async()=>{
    if(file.size>32*1024*1024)throw new Error('Import limit: 32 MB.');
    const data=JSON.parse(await file.text());const isRecording=data.format==='flybrain-dodge-recording';
    status(isRecording?'Recomputing recorded feedback…':'Restoring local checkpoint…');
    const result=await call(isRecording?'replay':'restore',isRecording?{recording:data}:{checkpoint:data});
    const checkpoint=isRecording?data.final:data;
    bundle={format:'flybrain-model-bundle',schema_version:1,model:checkpoint.brain.model,
      config:checkpoint.brain.config,model_sha256:checkpoint.model_sha256};
    importedModel=true;lastFrame=isRecording?(data.frames.at(-1)??null):null;draw();status(result.verified?'Replay verified. Every recorded frame and final state matched.':'Checkpoint restored. Run to continue from this state.');
    $('restore').value='';
  },{resume:false});
};
window.addEventListener('resize',draw);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
load('toy');
