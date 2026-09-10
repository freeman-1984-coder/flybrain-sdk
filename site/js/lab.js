import { bytes } from "./model-loader.js";
import { createDemo } from "./template.js";
const $ = id => document.getElementById(id);
const worker = new Worker(new URL('./lab-worker.js', import.meta.url), {type:'module'});
let nextId=0, pending=new Map(), bundle=null, modelId='toy', assetHash=null, running=false, busy=false;
let frame=null, history=[], groupIds={}, groupIndices={}, silenced=[], chartMax=60, loopTimer=null;
const colors=['#4263eb','#9b4edb','#ed9453'];
function call(op,data={}) {
  return new Promise((resolve,reject)=>{
    const id=++nextId;pending.set(id,{resolve,reject,op});worker.postMessage({id,op,...data});
  });
}
worker.onmessage=({data})=>{
  const p=pending.get(data.id);if(!p)return;pending.delete(data.id);
  if(data.ok){frame=data;if(p.op!=='load')draw();p.resolve(data);}else p.reject(new Error(data.error));
};
worker.onerror=()=>{for(const p of pending.values())p.reject(new Error('Simulation worker failed. Reload this page.'));pending.clear();fail(new Error('Simulation worker failed.'));};
function fail(error){running=false;$('run').textContent='Run';$('status').textContent=error.message;$('status').dataset.error='true';}
function status(text){$('status').textContent=text;delete $('status').dataset.error;}
function enabled(value){document.querySelectorAll('[data-needs-model]').forEach(b=>b.disabled=!value);}
function pause(){running=false;clearTimeout(loopTimer);loopTimer=null;$('run').textContent='Run';}

async function load(id){
  pause();enabled(false);$('load-real').disabled=true;
  status(id==='toy'?'Loading the offline toy…':'Downloading and verifying 3.8 MB…');
  try{
    let url=new URL('../toy-bundle.json',import.meta.url).href, size=null, hash=null;
    if(id!=='toy'){
      const response=await fetch(new URL('../model-catalog.json',import.meta.url));
      if(!response.ok)throw new Error('Cannot read the model catalog.');
      const catalog=await response.json();const entry=catalog.find(m=>m.id===id);
      if(!entry || entry.status!=='ready')throw new Error('Model is not available.');
      const asset=entry.assets.model;url=asset.url;size=asset.size_bytes;hash=asset.checksum.replace('sha256:','');
      if(!/^https:\/\//.test(url)||!/^[a-f0-9]{64}$/.test(hash))throw new Error('Invalid model integrity metadata.');
    }
    const next=JSON.parse(new TextDecoder().decode(await bytes(url,size,hash)));
    const result=await call('load',{bundle:next,modelId:id,assetHash:hash});
    bundle=next;modelId=id;assetHash=hash;frame=result;history=[];silenced=[];
    const model=bundle.model;
    $('model-name').textContent=id==='toy'?'Toy · artificial circuit':'MaleCNS · real anatomical subgraph';
    $('model-count').textContent=`${model.neuron_ids.length.toLocaleString()} neurons · ${model.synapses.length.toLocaleString()} edges · JavaScript CPU`;
    $('model-note').textContent=id==='toy'?'This 12-cell circuit is hand-designed. Load the real model to explore measured anatomy.':'Source wiring is real. Dynamics, input currents and the GF readout are assumed. This is not a complete biological fly.';
    $('stimulus').replaceChildren(...Object.keys(model.sensory).map(k=>new Option(k,k)));
    $('target').replaceChildren(new Option('All output neurons','output'),new Option('All input neurons','input'));
    const outputIds=[...new Set(Object.values(model.motor).flat())];
    for(const id of outputIds)$('target').add(new Option(`Cell ${id}`,id));
    groupIds=id==='toy'?{Input:[...new Set(Object.values(model.sensory).flat())],Relay:model.neuron_ids.filter(n=>n.startsWith('relay.')),Output:outputIds}:{LC4:model.neuron_ids.filter(n=>model.annotations[n].cell_type==='LC4'),LPLC2:model.neuron_ids.filter(n=>model.annotations[n].cell_type==='LPLC2'),GF:outputIds};
    groupIndices=Object.fromEntries(Object.entries(groupIds).map(([k,ids])=>[k,ids.map(n=>model.neuron_ids.indexOf(n))]));
    $('groups').replaceChildren(...Object.keys(groupIds).map((k,i)=>{
      const el=document.createElement('div');el.className='group';el.style.setProperty('--group',colors[i]);
      const title=document.createElement('span');title.textContent=`${k} · ${groupIds[k].length} cells`;
      const value=document.createElement('strong');value.id=`group-${k}`;value.textContent='0.0 Hz';el.append(title,value);return el;
    }));
    $('cells').replaceChildren(...model.neuron_ids.map(n=>new Option(`${n}${model.annotations?.[n]?.cell_type?' · '+model.annotations[n].cell_type:''}`,n)));
    enabled(true);status(id==='toy'?'Ready. Choose a stimulus, then run.':'Real model verified. It now runs locally in this browser.');draw();
  }catch(error){fail(error);enabled(!!bundle);}finally{$('load-real').disabled=false;}
}
function mean(indices,values){return indices.reduce((s,i)=>s+values[i],0)/indices.length;}
function canvas(id){const c=$(id),r=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1;c.width=Math.round(r.width*dpr);c.height=Math.round(r.height*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);return [ctx,r.width,r.height];}
function draw(){
  if(!frame||!bundle)return;
  $('time').textContent=`${frame.state.time_ms.toFixed(0)} ms`;
  $('events').textContent=`${frame.recordedCommands} recorded commands`;
  const means=Object.entries(groupIndices).map(([name,ids])=>{const m=mean(ids,frame.state.rates_hz);const el=$(`group-${name}`);if(el)el.textContent=`${m.toFixed(1)} Hz`;return m;});
  const id=$('cells').value, i=bundle.model.neuron_ids.indexOf(id);
  if(i>=0)$('cell-state').textContent=`${id}: voltage ${frame.state.voltage[i].toFixed(3)} · rate ${frame.state.rates_hz[i].toFixed(2)} Hz · ${frame.state.spikes[i]?'spike this tick':'no spike this tick'}`;
  $('action-values').textContent=Object.entries(frame.action).map(([k,v])=>`${k}: ${v.toFixed(3)}`).join('  ·  ');
  $('silenced').textContent=silenced.length?`${silenced.length} cells silenced`:'No cells silenced';
  const intensity=Math.max(0,...Object.values(frame.action));$('lamp').style.setProperty('--level',intensity);$('lamp-value').textContent=intensity.toFixed(3);
  const [ctx,w,h]=canvas('network');ctx.clearRect(0,0,w,h);
  const positions=[[w*.18,h*.5],[w*.50,h*.5],[w*.82,h*.5]];
  ctx.strokeStyle='#bac6df';ctx.lineWidth=2;
  const owner=new Map(Object.values(groupIds).flatMap((ids,k)=>ids.map(id=>[id,k])));
  const pairs=new Set(bundle.model.synapses.map(e=>`${owner.get(e.pre)},${owner.get(e.post)}`));
  for(const pair of pairs){const [a,b]=pair.split(',').map(Number);if(a===b||!positions[a]||!positions[b])continue;ctx.beginPath();ctx.moveTo(...positions[a]);ctx.quadraticCurveTo((positions[a][0]+positions[b][0])/2,h*.5+(a<b?-40:40),...positions[b]);ctx.stroke();}
  Object.entries(groupIds).forEach(([name,ids],k)=>{
    const [x,y]=positions[k];ctx.fillStyle=colors[k];ctx.globalAlpha=.2+.8*Math.min(1,(means[k]||0)/80);
    ctx.beginPath();ctx.arc(x,y,27,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
    ctx.fillStyle='#263656';ctx.textAlign='center';ctx.font='600 14px system-ui';ctx.fillText(name,x,y+52);
    ctx.font='12px system-ui';ctx.fillText(`${ids.length} cells`,x,y+70);
  });
  const [p,pw,ph]=canvas('history');p.clearRect(0,0,pw,ph);p.strokeStyle='#d9e0ee';
  for(let k=1;k<4;k++){p.beginPath();p.moveTo(36,ph*k/4);p.lineTo(pw,ph*k/4);p.stroke();}
  chartMax=Math.max(60,...history.flatMap(x=>x.means));
  p.fillStyle='#576781';p.font='11px system-ui';p.fillText(`${chartMax.toFixed(0)} Hz`,0,12);p.fillText('0',14,ph-4);
  for(let k=0;k<3;k++){p.beginPath();p.strokeStyle=colors[k];p.lineWidth=2;history.forEach((point,i)=>{
    const x=36+(pw-40)*i/Math.max(1,history.length-1),y=ph-6-(ph-22)*(point.means[k]||0)/chartMax;
    if(i)p.lineTo(x,y);else p.moveTo(x,y);
  });p.stroke();}
}
async function advance(ticks){
  const result=await call('advance',{ticks});
  history.push({tick:result.state.tick,means:Object.values(groupIndices).map(ids=>mean(ids,result.state.rates_hz))});
  if(history.length>250)history.shift();draw();
}
async function loop(){if(!running||busy)return;busy=true;try{await advance(20);}catch(e){fail(e);}finally{busy=false;if(running)loopTimer=setTimeout(loop,20);}}
async function guarded(action){try{await action();}catch(e){fail(e);}}
$('run').onclick=()=>{if(running)pause();else{running=true;$('run').textContent='Pause';loop();}};
$('step').onclick=()=>guarded(async()=>{pause();await advance(20);});
$('stimulate').onclick=()=>guarded(async()=>{await call('stimulate',{channel:$('stimulus').value,strength:Number($('strength').value),durationMs:Number($('duration').value)});status('Stimulus queued. Run or step to advance neural time.');});
$('reset').onclick=()=>guarded(async()=>{pause();await call('reset');history=[];silenced=[];draw();status('Reset to resting state. Recording cleared.');});
$('load-real').onclick=()=>load('male-cns-escape-v1');$('load-toy').onclick=()=>load('toy');
$('strength').oninput=()=>$('strength-value').textContent=Number($('strength').value).toFixed(1);
$('cells').onchange=draw;
$('silence').onclick=()=>guarded(async()=>{
  const t=$('target').value,m=bundle.model;
  const ids=t==='output'?[...new Set(Object.values(m.motor).flat())]:t==='input'?[...new Set(Object.values(m.sensory).flat())]:[t];
  const all=ids.every(id=>silenced.includes(id));await call('silence',{ids,enabled:!all});
  silenced=all?silenced.filter(id=>!ids.includes(id)):[...new Set([...silenced,...ids])];draw();
  status(all?'Selected cells released.':'New spikes suppressed. Previously emitted spikes still propagate; rate history decays.');
});
function download(name,text,type='application/json'){
 const link=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('export').onclick=()=>guarded(async()=>{pause();const result=await call('export');download('flybrain-experiment.json',JSON.stringify(result.experiment,null,2));status('Experiment exported. Replay it with the Python example linked below.');});
$('template').onclick=()=>guarded(async()=>{
  pause();const response=await fetch(new URL('./runtime.js',import.meta.url));if(!response.ok)throw new Error('Cannot load template runtime.');
  const runtime=await response.text();
  const licenseResponse=await fetch(new URL('../license.txt',import.meta.url));
  if(!licenseResponse.ok)throw new Error('Cannot load code license.');
  const template=createDemo(bundle,modelId,runtime,await licenseResponse.text());
  download('flybrain-demo.html',template,'text/html');status('Standalone demo exported with model and runtime. Open it locally and edit the marked section.');
});
window.addEventListener('resize',draw);
load('toy');
