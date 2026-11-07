const commands = {
  "http": "POST /v1/chat/completions\nAuthorization: Bearer <API_KEY>\nContent-Type: application/json\n\n{\n  \"messages\": [\n    {\"role\": \"user\", \"content\": \"Reply with OK.\"}\n  ],\n  \"stream\": true,\n  \"max_tokens\": 128,\n  \"temperature\": 0\n}",
  "generate": "{\"operation\":\"generate\",\"rendered_prompt\":\"<complete rendered chat template>\",\"context_tokens\":8192,\"max_new_tokens\":128,\"sampling\":{\"mode\":\"greedy\"},\"stream\":true,\"request_id\":1}",
  "score": "{\"operation\":\"score\",\"input_token_ids\":[1,2,3],\"candidate_token_ids\":[10,11],\"context_tokens\":8192,\"temperature\":1.0}",
  "convert": "stark convert Qwen/Qwen3-0.6B \\\n  --output models/qwen3"
};
const commandMetadata = {
  "http": {
    "heading": "HTTP / STREAMING CHAT",
    "detail": "Request shape · Successful SSE ends with [DONE]",
    "note": "Use the Base URL under Connections → Other apps. Omitting model targets the running model. Copy curl Example under Advanced settings generates an authenticated request that keeps the key out of command arguments and environment variables."
  },
  "generate": {
    "heading": "STDIN / GENERATE REQUEST",
    "detail": "One JSON line · text-generation-v1",
    "note": "Spawn stark generate --model-path /path/to/model and read readiness first. Supply a complete rendered chat template or input_token_ids. Consume progress/delta events and the terminal reply before the next request."
  },
  "score": {
    "heading": "STDIN / SCORE REQUEST",
    "detail": "One JSON line · candidate IDs are illustrative",
    "note": "Spawn stark score --score-model-path /path/to/model and read readiness first. Use token IDs from the loaded tokenizer. Returns logits, full-vocabulary log probabilities and temperature-scaled candidate probabilities."
  },
  "convert": {
    "heading": "SHELL / MODEL CONVERSION",
    "detail": "Installed CLI · selected checkpoint formats",
    "note": "Conversion prepares the artifact for Stark. Consult the converter catalog for family and precision support; runtime admission and numerical validation are separate from conversion."
  }
};
const tabs = [...document.querySelectorAll('[data-command]')];
const code = document.querySelector('#command-code');
const panel = document.querySelector('#command-panel');
const copyStatus = document.querySelector('#copy-status');
const selectCommand = (tab) => {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  }
  code.textContent = commands[tab.dataset.command];
  const metadata = commandMetadata[tab.dataset.command];
  document.querySelector('#command-heading').textContent = metadata.heading;
  document.querySelector('#command-detail').textContent = metadata.detail;
  document.querySelector('#command-note').textContent = metadata.note;
  panel.setAttribute('aria-labelledby', tab.id);
  copyStatus.textContent = '';
};
for (const tab of tabs) {
  tab.addEventListener('click', () => selectCommand(tab));
  tab.addEventListener('keydown', (event) => {
    const index = tabs.indexOf(tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length
      : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    selectCommand(tabs[next]);
    tabs[next].focus();
  });
}
for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    const status = document.getElementById(button.dataset.copyStatus);
    try {
      await navigator.clipboard.writeText(target.textContent);
      status.textContent = 'Copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Text selected. Use your keyboard to copy.';
    }
  });
}
const appTabs = [...document.querySelectorAll('[data-app-panel]')];
function selectApp(tab, updateHash = false) {
  for (const item of appTabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.dataset.appPanel).hidden = !selected;
  }
  document.querySelector('[data-app-support]').hidden = tab.dataset.appPanel === 'engine';
  if (updateHash) history.replaceState(null, '', '#' + tab.dataset.appPanel);
}
for (const tab of appTabs) {
  tab.addEventListener('click', () => selectApp(tab, true));
  tab.addEventListener('keydown', (event) => {
    const index = appTabs.indexOf(tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % appTabs.length
      : event.key === 'ArrowLeft' ? (index + appTabs.length - 1) % appTabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? appTabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    selectApp(appTabs[next], true);
    appTabs[next].focus();
  });
}
function selectLinkedApp(initial = false) {
  const target = document.getElementById(location.hash.slice(1));
  const panelId = target?.closest('.app-guide')?.id;
  const linked = appTabs.find(tab => tab.dataset.appPanel === panelId);
  if (linked || initial) selectApp(linked || appTabs[0]);
  // A hidden panel cannot receive the browser's original fragment scroll.
  if (linked && !initial) document.getElementById(panelId).scrollIntoView({ block: 'start' });
}
if (appTabs.length) {
  selectLinkedApp(true);
  window.addEventListener('hashchange', () => selectLinkedApp());
}

const modelTabs = [...document.querySelectorAll('[data-model-group]')];
function selectModelGroup(tab) {
  for (const item of modelTabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.dataset.modelGroup).hidden = !selected;
  }
}
for (const tab of modelTabs) {
  tab.addEventListener('click', () => selectModelGroup(tab));
  tab.addEventListener('keydown', (event) => {
    const index = modelTabs.indexOf(tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % modelTabs.length
      : event.key === 'ArrowLeft' ? (index + modelTabs.length - 1) % modelTabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? modelTabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    selectModelGroup(modelTabs[next]);
    modelTabs[next].focus();
  });
}
function selectLinkedModelGroup() {
  const linked = modelTabs.find(tab => '#' + tab.dataset.modelGroup === location.hash);
  selectModelGroup(linked || modelTabs[0]);
}
if (modelTabs.length) {
  selectLinkedModelGroup();
  window.addEventListener('hashchange', selectLinkedModelGroup);
}
const engineRows = [...document.querySelectorAll('[data-engine-rates]')];
const workloadSelect = document.querySelector('#benchmark-workload');
const viewButtons = [...document.querySelectorAll('[data-view]')];
const rates = new Map(engineRows.map(row => [row, JSON.parse(row.dataset.engineRates)]));
let relativeView = false;
function renderBenchmark() {
  const index = Number(workloadSelect.value);
  const baselineRow = engineRows.find(row => row.dataset.engineId === 'mlx_lm');
  const baseline = rates.get(baselineRow)[index];
  const maximum = Math.max(...engineRows.map(row => rates.get(row)[index]));
  for (const row of engineRows) {
    const rate = rates.get(row)[index];
    row.querySelector('.engine-value').textContent = relativeView
      ? (rate / baseline).toFixed(3) + '×' : rate.toFixed(1);
    row.querySelector('.bar').style.setProperty('--bar', (rate / maximum * 100) + '%');
  }
  for (const button of viewButtons) {
    const selected = (button.dataset.view === 'relative') === relativeView;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-pressed', String(selected));
  }
  document.querySelector('#rate-heading span').textContent =
    relativeView ? 'MLX-LM = 1.000×' : 'tokens / second';
  const label = workloadSelect.selectedOptions[0].textContent;
  document.querySelector('#chart-selection').textContent =
    index === 0 ? 'Overall · geometric mean of six workloads' : label + ' · four independent model loads';
  document.querySelector('#benchmark-caption').textContent =
    label + ': Qwen3.5-2B generation speed on Apple M5 Max.';
}
workloadSelect?.addEventListener('change', renderBenchmark);
for (const button of viewButtons) {
  button.addEventListener('click', () => {
    relativeView = button.dataset.view === 'relative';
    renderBenchmark();
  });
}
if (workloadSelect) renderBenchmark();

const benchmarkTabs = [...document.querySelectorAll('[data-benchmark-panel]')];
function selectBenchmarkModel(tab, updateHash = false) {
  for (const item of benchmarkTabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.dataset.benchmarkPanel).hidden = !selected;
  }
  if (updateHash) history.replaceState(null, '', '#' + tab.dataset.benchmarkPanel);
}
for (const tab of benchmarkTabs) {
  tab.addEventListener('click', () => selectBenchmarkModel(tab, true));
  tab.addEventListener('keydown', (event) => {
    const index = benchmarkTabs.indexOf(tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % benchmarkTabs.length
      : event.key === 'ArrowLeft' ? (index + benchmarkTabs.length - 1) % benchmarkTabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? benchmarkTabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    selectBenchmarkModel(benchmarkTabs[next], true);
    benchmarkTabs[next].focus();
  });
}
function selectLinkedBenchmark(initial = false) {
  const target = document.getElementById(location.hash.slice(1));
  const panelId = target?.closest('.benchmark-panel')?.id;
  const linked = benchmarkTabs.find(tab => tab.dataset.benchmarkPanel === panelId);
  if (linked || initial || location.hash === '#benchmarks') {
    selectBenchmarkModel(linked || benchmarkTabs[0]);
  }
}
if (benchmarkTabs.length) {
  selectLinkedBenchmark(true);
  window.addEventListener('hashchange', () => selectLinkedBenchmark());
}
