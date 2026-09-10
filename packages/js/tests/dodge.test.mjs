import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {DodgeSession} from '../dist/dodge.js';
const json=p=>JSON.parse(readFileSync(new URL(p,import.meta.url),'utf8'));
const toy=json('../../../site/toy-bundle.json'),real=json('../../../models/male-cns-escape-v1/model.json');
function near(a,b){
 if(typeof a==='number'){assert.ok(Math.abs(a-b)<=1e-9,`${a} != ${b}`);return;}
 if(a&&typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort());for(const k of Object.keys(a))near(a[k],b[k]);}
 else assert.equal(a,b);
}
for(const trial of json('./fixtures/dodge-reference.json'))test(`Live game agrees with Python feedback: ${trial.model}`,()=>{
 const session=new DodgeSession(trial.model==='toy'?toy:real);
 for(const expected of trial.trace){
  for(const event of trial.events)if(event.frame===session.frameIndex)session.command(event.command);
  near(session.step(),expected);
 }
 const state=session.save().brain.state;
 for(const key of ['voltage','spikes','rates_hz','refractory','silenced'])near(state[key],trial.final_state[key]);
});
for(const [name,bundle]of[['toy',toy],['real',real]])test(`Checkpoints, edits and full replay: ${name}`,()=>{
 let a=new DodgeSession(bundle);a.command({op:'add',x:.3,y:.7});
 for(let i=0;i<18;i++)a.step();
 a.command({op:'silence',enabled:true});a.step();
 const b=DodgeSession.restore(JSON.parse(JSON.stringify(a.save())));
 for(let i=0;i<35;i++)assert.deepEqual(a.step(),b.step());
 a.command({op:'silence',enabled:false}); // event after final step must also be replayed
 const recording=a.recording();assert.deepEqual(DodgeSession.replay(recording).save(),a.save());
 for(const mutate of [r=>r.frames[2].applied.steer=99,r=>r.events[0].frame=12,
  r=>r.frames[0].brain_tick+=1e-10,r=>r.final.input_gain=0]){
  const bad=structuredClone(recording);mutate(bad);assert.throws(()=>DodgeSession.replay(bad));
 }
 assert.deepEqual(DodgeSession.replay(b.recording()).save(),b.save());
});
test('Live obstacle inputs affect neural control; silencing removes new drive',()=>{
 const live=new DodgeSession(real),silent=new DodgeSession(real);
 for(const s of [live,silent]){s.command({op:'clear'});s.command({op:'add',x:.3,y:.7});}
 silent.command({op:'silence',enabled:true});
 for(let i=0;i<10;i++){live.step();silent.step();}
 assert.ok(live.environment.x>.5);assert.equal(silent.environment.x,.5);
 assert.ok(live.outputRates[0]>0);assert.deepEqual(silent.outputRates,[0,0]);
});
test('Invalid commands/checkpoints are rejected without partial edits',()=>{
 const s=new DodgeSession(toy),before=s.save();
 for(const command of [{op:'add',x:NaN,y:.5},{op:'configure',input_gain:2,output_gain:99},
  {op:'silence',enabled:1},{op:'unknown'}]){assert.throws(()=>s.command(command));assert.deepEqual(s.save(),before);}
 for(const mutate of [d=>d.environment.frame=1,d=>d.silenced=true,d=>d.environment.rng=-1,
  d=>d.environment.blocks[0].y=NaN,d=>d.schema_version=2]){
  const data=structuredClone(before);mutate(data);assert.throws(()=>DodgeSession.restore(data));
 }
});
