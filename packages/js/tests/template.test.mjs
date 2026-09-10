import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createDemo } from '../../../site/js/template.js';
const read = path => readFileSync(new URL(path,import.meta.url),'utf8');
for(const model of ['toy','real']) test(`exported standalone HTML executes without network: ${model}`,()=>{
 const bundle=JSON.parse(read(model==='toy'?'../../../site/toy-bundle.json':'../../../models/male-cns-escape-v1/model.json'));
 const html=createDemo(bundle,model==='toy'?'toy':'male-cns-escape-v1',read('../dist/index.js'),read('../../../LICENSE'));
 assert.ok(html.includes('Copyright (c) 2026 flybrain-sdk contributors'));
 assert.ok(html.includes('EDIT HERE'));
 assert.equal((html.match(/<script/g)||[]).length,1);
 const source=html.match(/<script type="module">([\s\S]*)<\/script>/)[1];
 const nodes=Object.fromEntries(['input','step','reset','output'].map(id=>[id,{textContent:''}]));
 // No network, file, or Node globals are provided to the exported runtime.
 vm.runInNewContext(source,{document:{getElementById:id=>nodes[id]}},{timeout:5000});
 assert.equal(JSON.parse(nodes.output.textContent).time_ms,0);
 nodes.input.onclick();nodes.step.onclick();
 const state=JSON.parse(nodes.output.textContent);
 assert.equal(state.time_ms,100);assert.ok(Math.max(...Object.values(state.action))>0);
 nodes.reset.onclick();assert.equal(JSON.parse(nodes.output.textContent).time_ms,0);
});
