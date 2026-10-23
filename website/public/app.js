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
