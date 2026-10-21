const commands = {
  generate: 'cargo +1.98.1 build --release --locked \\\n  -p stark-generate\n\n./target/release/stark-generate \\\n  --model-path /path/to/converted-model',
  score: 'cargo +1.98.1 build --release --locked \\\n  -p stark-score --features optimized\n\n./target/release/stark-score \\\n  --score-model-path /path/to/converted-model',
  convert: 'uv sync --project tools/converter --locked\n\nuv run --project tools/converter --locked \\\n  stark-convert Qwen/Qwen3-0.6B \\\n  --output models/qwen3',
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
document.querySelector('#copy-command').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(code.textContent);
    copyStatus.textContent = 'Commands copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Commands selected. Use your keyboard to copy.';
  }
});
for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => {
    const relative = button.dataset.view === 'relative';
    for (const item of document.querySelectorAll('[data-view]')) {
      const selected = item === button;
      item.classList.toggle('active', selected);
      item.setAttribute('aria-pressed', String(selected));
    }
    for (const row of document.querySelectorAll('[data-stark]')) {
      const stark = Number(row.dataset.stark);
      const mlx = Number(row.dataset.mlx);
      // Relative values use the recorded paired gain, rather than rounded rates.
      row.querySelector('.stark-value').textContent = relative
        ? (1 + Number(row.dataset.gain) / 100).toFixed(3) + '×' : stark.toFixed(1);
      row.querySelector('.mlx-value').textContent = relative ? '1.000×' : mlx.toFixed(1);
      row.querySelector('.mlx-bar').style.setProperty('--bar',
        (relative ? 100 / (1 + Number(row.dataset.gain) / 100) : mlx / stark * 100) + '%');
    }
    for (const id of ['stark-heading', 'mlx-heading']) {
      document.querySelector('#' + id + ' span').textContent = relative ? 'MLX-LM = 1.000×' : 'tokens / second';
    }
  });
}

const benchmarkTabs = [...document.querySelectorAll('[data-benchmark]')];
const selectBenchmark = (tab) => {
  for (const item of benchmarkTabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    const section = document.getElementById(item.dataset.benchmark);
    section.hidden = !selected;
    section.setAttribute('role', 'tabpanel');
    section.setAttribute('aria-labelledby', item.id);
    section.tabIndex = 0;
  }
};
for (const tab of benchmarkTabs) {
  tab.addEventListener('click', () => selectBenchmark(tab));
  tab.addEventListener('keydown', (event) => {
    const index = benchmarkTabs.indexOf(tab);
    const next = event.key === 'ArrowRight' ? (index + 1) % benchmarkTabs.length
      : event.key === 'ArrowLeft' ? (index + benchmarkTabs.length - 1) % benchmarkTabs.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? benchmarkTabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    selectBenchmark(benchmarkTabs[next]);
    benchmarkTabs[next].focus();
  });
}
selectBenchmark(benchmarkTabs.find((tab) => '#' + tab.dataset.benchmark === window.location.hash) || benchmarkTabs[0]);
document.querySelector('.benchmark-tabs').hidden = false;

const workload = document.querySelector('#stock-workload');
const stockRows = [...document.querySelectorAll('[data-rates]')];
const stockRates = stockRows.map((row) => row.dataset.rates.split(',').map(Number));
workload.disabled = false;
workload.addEventListener('change', () => {
  const index = Number(workload.value);
  const highest = Math.max(...stockRates.map((rates) => rates[index]));
  stockRows.forEach((row, rowIndex) => {
    const rate = stockRates[rowIndex][index];
    row.querySelector('.stock-value').textContent = rate.toFixed(1);
    row.querySelector('.comparison-bar').style.setProperty('--bar', rate / highest * 100 + '%');
  });
  document.querySelector('#stock-caption').textContent =
    workload.selectedOptions[0].textContent + ' — complete-request output tokens per second';
});
