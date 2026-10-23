const commands = {
  chat: 'stark chat \\\n  --model-path /path/to/converted-model',
  generate: 'stark generate \\\n  --model-path /path/to/converted-model',
  score: 'stark score \\\n  --score-model-path /path/to/converted-model',
  convert: 'stark convert Qwen/Qwen3-0.6B \\\n  --output models/qwen3',
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
      status.textContent = 'Commands copied.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Commands selected. Use your keyboard to copy.';
    }
  });
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
workloadSelect.addEventListener('change', renderBenchmark);
for (const button of viewButtons) {
  button.addEventListener('click', () => {
    relativeView = button.dataset.view === 'relative';
    renderBenchmark();
  });
}
renderBenchmark();
