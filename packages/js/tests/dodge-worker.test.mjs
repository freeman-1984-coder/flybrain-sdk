import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
let reply;
globalThis.self={postMessage:value=>{reply=value;}};
await import('../../../site/js/dodge-worker.js');
const bundle=JSON.parse(readFileSync(new URL('../../../models/male-cns-escape-v1/model.json',import.meta.url)));
function call(op,data={}){self.onmessage({data:{id:42,op,...data}});assert.equal(reply.id,42);return reply;}
test('Actual site worker saves, restores, verifies and preserves state after invalid imports',()=>{
 assert.equal(call('load',{bundle}).ok,true);
 assert.equal(call('command',{command:{op:'add',x:.3,y:.7}}).ok,true);
 const step=call('step');assert.ok(step.frame.applied.steer>0);
 const checkpoint=call('save').checkpoint;
 const expected=call('step').frame;
 assert.equal(call('restore',{checkpoint}).ok,true);
 assert.deepEqual(call('step').frame,expected);
 const recording=call('export').recording;
 assert.equal(call('replay',{recording}).verified,true);
 const before=call('save').checkpoint;
 const bad=structuredClone(recording);bad.frames[0].applied.steer=999;
 assert.equal(call('replay',{recording:bad}).ok,false);
 assert.deepEqual(call('save').checkpoint,before);
});
