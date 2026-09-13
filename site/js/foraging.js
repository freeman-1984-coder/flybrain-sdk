'use strict';
const $=id=>document.getElementById(id), canvas=$('world'), ctx=canvas.getContext('2d');
let data=null, run=null, playing=false, elapsed=0, last=performance.now(), angle=.65, drag=null;
const colors={top:'#344b49',left:'#253c3e',right:'#1d3037'};
function project(x,y,z){const dx=x+1,dz=z+1,c=Math.cos(angle),s=Math.sin(angle);return [500+(dx*c-dz*s)*38,230+(dx*s+dz*c)*18-y*39]}
function polygon(points,color){ctx.fillStyle=color;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill()}
function cube(x,z){const p=(dx,y,dz)=>project(x+dx,y,z+dz),sx=Math.sin(angle)>=0?.5:-.5,sz=Math.cos(angle)>=0?.5:-.5;polygon([p(-.5,0,sz),p(.5,0,sz),p(.5,1,sz),p(-.5,1,sz)],colors.left);polygon([p(sx,0,-.5),p(sx,0,.5),p(sx,1,.5),p(sx,1,-.5)],colors.right);polygon([p(-.5,1,-.5),p(.5,1,-.5),p(.5,1,.5),p(-.5,1,.5)],colors.top)}
function draw(){
 ctx.clearRect(0,0,1000,520);ctx.fillStyle='#101d25';ctx.fillRect(0,0,1000,520);
 for(let x=-7;x<7;x++)for(let z=-7;z<7;z++)polygon([project(x,0,z),project(x+1,0,z),project(x+1,0,z+1),project(x,0,z+1)],(x+z)%2?'#1c302f':'#213633');
 if(!run)return;
 const index=Math.max(0,Math.min(run.frames.length-1,Math.floor(elapsed/20)-1));
 const frame=elapsed<20?run.initial:run.frames[index];
 const blocks=[...frame.blocks].sort((a,b)=>(a[0]*Math.sin(angle)+a[1]*Math.cos(angle))-(b[0]*Math.sin(angle)+b[1]*Math.cos(angle)));
 blocks.forEach(b=>cube(...b));
 ctx.strokeStyle='#86efc0';ctx.lineWidth=2;ctx.beginPath();
 [run.initial,...run.frames.slice(0,elapsed<20?0:index+1)].forEach((v,i)=>{const p=project(v.position[0],.03,v.position[2]);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.stroke();
 const food=project(...frame.food);ctx.font='32px system-ui';ctx.textAlign='center';ctx.fillText('🍌',food[0],food[1]);ctx.font='12px system-ui';ctx.fillStyle='#f7d183';ctx.fillText('食物气味源',food[0],food[1]-33);
 const [fx,fy,fz]=frame.position, at=project(fx,fy,fz), head=project(fx+.23*Math.cos(frame.yaw),fy,fz+.23*Math.sin(frame.yaw));
 ctx.fillStyle='#7baac777';ctx.beginPath();ctx.ellipse(at[0]-7,at[1]-7,13,5,-.4,0,Math.PI*2);ctx.ellipse(at[0]+7,at[1]-7,13,5,.4,0,Math.PI*2);ctx.fill();
 ctx.strokeStyle='#e1e8d2';ctx.lineWidth=8;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(...at);ctx.lineTo(...head);ctx.stroke();ctx.lineCap='butt';
 ctx.fillStyle='#ffffff';ctx.beginPath();ctx.arc(head[0],head[1],4,0,Math.PI*2);ctx.fill();
 ctx.textAlign='left';ctx.font='14px system-ui';ctx.fillStyle='#b3c9c7';ctx.fillText(run.name==='neural_control'?'完整脑神经输出 → 身体':'关闭嗅觉输入 · 完整脑仍在 GPU 演算',25,32);
 $('time').textContent=`${Math.round(elapsed/20)*20} / ${data.protocol.duration_ms} ms`;$('seek').value=elapsed;
 const rates=frame.output_rates_hz||[0,0],action=frame.action||{speed:0,turn:0};
 $('neural').textContent=`DNa02 左 / 右：${rates.map(v=>v.toFixed(1)+' Hz').join(' / ')}`;
 $('motor').textContent=`速度 ${action.speed.toFixed(2)} 单位/秒 · 转向 ${action.turn.toFixed(2)} 弧度/秒`;
 $('odor').textContent=frame.odor?`触角浓度 左 ${frame.odor.odor_left.toFixed(3)} / 右 ${frame.odor.odor_right.toFixed(3)}（无量纲）`:'尚未施加第一帧刺激。';
}
function select(index){run=data.runs[index];elapsed=0;playing=false;$('play').textContent='播放';$('result').textContent=run.reached_food?'已接触':'未接触';$('distance').textContent=run.final_distance.toFixed(2);$('summary').textContent=`初始距离 ${run.initial_distance.toFixed(2)} → 最终 ${run.final_distance.toFixed(2)}。${run.reached_food?'本次记录接触了食物；单次结果不能证明可靠导航。':'本次记录未接触食物。'}`;$('status').textContent=`${data.device.name} 实际录制 · 此组模拟 2 秒，耗时 ${run.wall_seconds.toFixed(1)} 秒。`;draw()}
$('play').onclick=()=>{if(elapsed>=data.protocol.duration_ms)elapsed=0;playing=!playing;$('play').textContent=playing?'暂停':'播放'};
$('restart').onclick=()=>{elapsed=0;draw()};$('seek').oninput=e=>{elapsed=Number(e.target.value);playing=false;$('play').textContent='播放';draw()};$('condition').onchange=e=>select(Number(e.target.value));
canvas.onpointerdown=e=>{drag=e.clientX;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(drag!==null){angle+=(e.clientX-drag)*.008;drag=e.clientX;draw()}};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
function animate(now){if(playing&&data){elapsed=Math.min(data.protocol.duration_ms,elapsed+Math.min(now-last,100)*Number($('speed').value));if(elapsed>=data.protocol.duration_ms){playing=false;$('play').textContent='重播'}draw()}last=now;requestAnimationFrame(animate)}
draw();requestAnimationFrame(animate);
fetch('experiments/fullbrain-voxel.json').then(r=>{if(!r.ok)throw new Error('尚无完整实验记录');return r.json()}).then(value=>{
 if(value.schema!=='flybrain-voxel-recording-v1'||value.status!=='completed'||value.model.neurons!==139255||value.runs.length!==2||value.runs.some(r=>r.frames.length!==100))throw new Error('记录不完整，暂停回放');
 data=value;$('neurons').textContent=data.model.neurons.toLocaleString();$('condition').replaceChildren(...data.runs.map((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=r.name==='neural_control'?'正常嗅觉 → 神经控制':'对照：关闭嗅觉入口';return option}));
 for(const id of ['condition','play','restart','seek'])$(id).disabled=false;
 $('provenance').textContent=`${data.model.neurons.toLocaleString()} neurons · ${data.model.retained_rows.toLocaleString()} edges · seed ${data.seed} · ${data.started_utc} · ${data.model.graph_arrays_sha256.slice(0,16)}…`;
 select(0);
}).catch(error=>{$('status').textContent=`无法播放：${error.message}。页面不会用生成轨迹代替 GPU 记录。`;$('status').classList.add('error')});
