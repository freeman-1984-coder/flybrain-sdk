import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
let reply;
globalThis.self = {postMessage: value => {reply=value;}};
await import('../../../site/js/lab-worker.js');
const read = path => JSON.parse(readFileSync(new URL(path,import.meta.url),'utf8'));
function call(op,data={}) {
 self.onmessage({data:{id:1,op,...data}});
 assert.equal(reply.ok,true,reply.error);return reply;
}
for(const modelId of ['toy','male-cns-escape-v1']) test(`worker export: ${modelId}`,()=>{
 const bundle = read(modelId==='toy'?'../../../site/toy-bundle.json':'../../../models/male-cns-escape-v1/model.json');
 const assetHash=modelId==='toy'?null:read('../../../site/model-catalog.json').find(m=>m.id===modelId).assets.model.checksum.slice(7);
 call('load',{bundle,modelId,assetHash});
 call('stimulate',{channel:Object.keys(bundle.model.sensory)[0],strength:1,durationMs:150});
 call('advance',{ticks:21});
 const ids=Object.values(bundle.model.motor).flat();
 call('silence',{ids,enabled:true});call('advance',{ticks:19});
 call('silence',{ids,enabled:false});call('advance',{ticks:61});
 const recording=call('export').experiment;
 assert.equal(recording.duration_ticks,101);assert.equal(recording.commands.length,3);
 assert.ok(Math.max(...Object.values(recording.expected.action))>0);
 // Python's tests replay these actual worker exports, including a pending pulse.
 writeFileSync(new URL(`../../../tests/fixtures/browser-${modelId}.json`,import.meta.url),JSON.stringify(recording,null,2)+'\n');
 call('reset');assert.equal(call('export').experiment.commands.length,0);
 assert.ok(reply.state.voltage.every(v=>v===0));
});
