/* Recorded evidence only. Never synthesize data when the GPU report is missing. */
const $ = (id) => document.getElementById(id);
const sideName = (side) => side === 'left' ? '左' : '右';
const label = (run) => `${run.contact_mv} mV · ${run.order.map(sideName).join(' → ')}`;
const rateGroups = ['DNa02_left', 'DNa02_right', 'ALPN', 'MBON', 'descending'];
let report;
function row(parent, values) {
  const tr = document.createElement('tr');
  values.forEach((value) => {
    const td = document.createElement('td');
    td.textContent = String(value);
    tr.append(td);
  });
  parent.append(tr);
}
function svgNode(name, attributes, text) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attributes).forEach(([key, value]) => el.setAttribute(key, String(value)));
  if (text !== undefined) el.textContent = text;
  return el;
}
function render() {
  if (!report) return;
  const run = report.runs[Number($('run').value)];
  const population = $('population').value;
  const groups = [['DNa02_left', '#93e8c2'], ['DNa02_right', '#e8b9ff'], [population, '#ffc778']];
  const chart = $('trace');
  chart.replaceChildren(svgNode('title', {}, `${label(run)}：DNa02 左右和 ${population} 平滑平均频率`));
  chart.setAttribute('aria-label', `${label(run)} 的真实 GPU 神经活动，横轴 0–800 毫秒`);
  // Fixed across both conditions and population choices, so selection cannot exaggerate differences.
  let maximum = 10;
  for (const item of report.runs) for (const frame of item.trace) {
    for (const key of rateGroups) {
      maximum = Math.max(maximum, frame.mean_rates_hz[key]);
    }
  }
  maximum = Math.ceil(maximum / 20) * 20;
  const x = (ms) => 65 + ms / 800 * 910;
  const y = (rate) => 320 - rate / maximum * 270;
  for (const phase of run.phases.filter((p) => p.side !== null)) {
    const start = phase.start_tick * report.config.dt_ms;
    chart.append(svgNode('rect', {x: x(start), y: 40, width: phase.duration_ms / 800 * 910,
      height: 280, fill: phase.side === 'left' ? '#93e8c2' : '#e8b9ff', opacity: 0.09}));
    chart.append(svgNode('text', {x: x(start) + 8, y: 30, fill: '#aec1c2', 'font-size': 15}, `${sideName(phase.side)}侧气味`));
  }
  for (let tick = 0; tick <= 4; tick++) {
    const rate = maximum * tick / 4;
    chart.append(svgNode('line', {x1: 65, x2: 975, y1: y(rate), y2: y(rate), stroke: '#2a3e43'}));
    chart.append(svgNode('text', {x: 53, y: y(rate) + 5, fill: '#aec1c2', 'text-anchor': 'end', 'font-size': 14}, rate.toFixed(0)));
  }
  for (let ms = 0; ms <= 800; ms += 100) {
    chart.append(svgNode('text', {x: x(ms), y: 346, fill: '#aec1c2', 'text-anchor': 'middle', 'font-size': 14}, ms));
  }
  chart.append(svgNode('text', {x: 15, y: 30, fill: '#aec1c2', 'font-size': 14}, 'Hz'));
  chart.append(svgNode('text', {x: 980, y: 366, fill: '#aec1c2', 'text-anchor': 'end', 'font-size': 14}, '模拟时间 / ms'));
  for (const [key, color] of groups) {
    chart.append(svgNode('polyline', {points: run.trace.map((f) => `${x(f.time_ms)},${y(f.mean_rates_hz[key])}`).join(' '),
      fill: 'none', stroke: color, 'stroke-width': 2,
      'stroke-dasharray': key === 'DNa02_right' ? '6 4' : 'none'}));
  }
  $('timing').replaceChildren();
  run.summary.stimuli.forEach((stimulus) => row($('timing'), [
    sideName(stimulus.side), stimulus.onset_ipsilateral_minus_contralateral_hz,
    stimulus.later_ipsilateral_minus_contralateral_hz, stimulus.total_right_minus_left_spikes,
  ]));
}
async function load() {
  try {
    const response = await fetch('experiments/odor-gain-pilot.json');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.schema !== 'flybrain-odor-gain-pilot-v1' || data.status !== 'completed' ||
      data.runs?.length !== 8 || data.models?.length !== 4 ||
      data.models.some((m) => m.neurons !== 139255 || m.retained_rows !== 16847997) ||
      data.config?.dt_ms !== 0.1 ||
      data.runs.some((r, i) => r.phases?.length !== 5 || r.trace?.length !== 80 ||
        r.contact_mv !== [0.05, 0.10, 0.175, 0.275][Math.floor(i / 2)] ||
        r.order?.join(',') !== (i % 2 === 0 ? 'left,right' : 'right,left') ||
        r.graph_sha256 !== data.models[Math.floor(i / 2)].graph_arrays_sha256 ||
        r.trace.some((f, j) => Math.abs(f.time_ms - (j + 1) * 10) > 1e-6 ||
          rateGroups.some((key) => !Number.isFinite(f.mean_rates_hz?.[key]) || f.mean_rates_hz[key] < 0)) ||
        r.summary?.stimuli?.length !== 2 || r.summary?.recovery?.length !== 2)) {
      throw new Error('完整八组 GPU 记录尚未通过结构检查');
    }
    report = data;
    $('results').replaceChildren();
    report.runs.forEach((run, index) => {
      const option = document.createElement('option');
      option.value = index; option.textContent = label(run); $('run').append(option);
      row($('results'), [label(run),
        ...run.summary.stimuli.map((s) => s.onset_ipsilateral_minus_contralateral_hz),
        ...run.summary.recovery.map((r) => r.new_tail_spikes.descending)]);
    });
    $('run').disabled = false;
    $('run').addEventListener('change', render);
    $('population').addEventListener('change', render);
    $('status').textContent = `${report.device.name} 实际运行 · 八组全部完成 · 仅为单随机种子探索，未自动选择参数。`;
    $('provenance').textContent = `FlyWire v783 · 139,255 neurons · 16,847,997 edges · seed ${report.protocol.seed} · ${report.started_utc} · CuPy ${report.device.cupy}`;
    render();
  } catch (error) {
    $('status').textContent = `无法读取完整实验记录：${error.message}。页面不会用示意数据替代。`;
    $('run').disabled = true;
  }
}
load();
