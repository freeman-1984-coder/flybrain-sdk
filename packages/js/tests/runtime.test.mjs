import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { FlyBrain } from '../dist/index.js';
const json = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const toy = json('../../../src/flybrain/data/toy.json');
const real = json('../../../models/male-cns-escape-v1/model.json');
const trials = json('./fixtures/python-reference.json');
const near = (a,b,label) => assert.ok(Math.abs(a-b) <= 1e-10, `${label}: ${a} != ${b}`);
function command(brain,c) {
 if(c.op==='stimulate') brain.stimulate({channel:c.channel,strength:c.strength,durationMs:c.duration_ms});
 if(c.op==='silence') brain.silence(c.ids,c.enabled);
 if(c.op==='inject') brain.inject(c.ids,{amplitude:c.amplitude,durationMs:c.duration_ms});
}
for (const trial of trials) test(`Python parity every cell/tick: ${trial.model}`,()=>{
 let brain=FlyBrain.load(trial.model==='toy'?toy:real,{config:trial.config});
 for(const expected of trial.trace){
  for(const c of trial.commands) if(c.tick===brain.tick) command(brain,c);
  const state=brain.step();
  assert.equal(state.tick,expected.tick);assert.deepEqual(state.spikes,expected.spikes);
  state.voltage.forEach((v,i)=>near(v,expected.voltage[i],'voltage'));
  state.rates_hz.forEach((v,i)=>near(v,expected.rates_hz[i],'rate'));
  for(const [name,v] of Object.entries(brain.action())) near(v,expected.action[name],name);
  // Restore mid-pulse / during intervention as well as after expiry.
  if([30,120,180].includes(brain.tick)) brain=FlyBrain.restore(JSON.parse(JSON.stringify(brain.save())));
 }
});
test('selected readout, detached state, annotation IDs and checkpoint',()=>{
 const brain=FlyBrain.load(real);const ids=brain.select({cell_type:'DNp01'});
 assert.deepEqual(ids,['10001','10010']);
 brain.bindReadout({flash:ids}).inject(ids,{amplitude:2,durationMs:50});brain.advance(20);
 const observation=brain.observe(ids);assert.ok(brain.action().flash>0);
 observation.rates_hz[0]=999;
 const restored=FlyBrain.restore(brain.save());
 assert.deepEqual(brain.step(50),restored.step(50));assert.deepEqual(brain.action(),restored.action());
 assert.notEqual(brain.observe(ids).rates_hz[0],999);
});
test('silencing a relay removes downstream output, release recovers',()=>{
 const brain=FlyBrain.load(toy);brain.silence(['relay.food']);
 brain.stimulate({channel:'food',durationMs:200});brain.advance(100);
 assert.equal(brain.action().walk,0);brain.silence(['relay.food'],false);brain.advance(100);
 assert.ok(brain.action().walk>0);
});
test('invalid inputs leave state intact and unavailable backends fail',()=>{
 const b=FlyBrain.load(toy);const before=b.save();
 for(const operation of [()=>b.step(0),()=>b.step(1.5),()=>b.inject(['missing'],{amplitude:1,durationMs:1}),
  ()=>b.stimulate({channel:'food',strength:2}),()=>b.inject(['sense.food'],{amplitude:1,durationMs:.5}),
  ()=>b.bindReadout({flash:['missing']}),()=>b.silence(['sense.food'],'yes')]) assert.throws(operation);
 assert.deepEqual(b.save(),before);
 for(const backend of ['wasm','cuda']) assert.throws(()=>FlyBrain.load(toy,{backend}),/not implemented/);
});
test('checkpoint rejects malformed vectors and nonfinite state',()=>{
 for(const mutate of [d=>d.state.voltage.pop(),d=>d.state.rates_hz[0]=-1,d=>d.state.spikes[0]=1,
  d=>d.state.silenced[0]=0,d=>d.state.voltage[0]=NaN,d=>d.state.refractory[0]=999,
  d=>d.pending_currents=[{ids:['missing'],amplitude:1,remaining_steps:1}],d=>d.schema_version=2]){
  const d=FlyBrain.load(toy).save();mutate(d);assert.throws(()=>FlyBrain.restore(d));
 }
});
