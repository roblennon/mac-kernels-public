import { app } from '/scripts/app.js';
import { captureRecipe, refreshChoices, useExport, isExportActive, exportMessage } from './export_helpers.mjs';

async function request(path, options) {
  const response = await fetch(`/krea-workbench/${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error ?? `Request failed (${response.status})`);
  return body;
}
function post(path, body) {
  return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}
function element(tag, text, parent) {
  const item = document.createElement(tag);
  if (text) item.textContent = text;
  parent?.append(item);
  return item;
}

async function openExport(node) {
  if (document.querySelector('#krea-checkpoint-export')) return;
  const recipe = captureRecipe(node);
  const dialog = element('dialog');
  dialog.id = 'krea-checkpoint-export';
  dialog.className = 'comfy-modal';
  Object.assign(dialog.style, { display: 'block', width: 'min(480px, calc(100vw - 32px))',
    boxSizing: 'border-box', color: 'var(--fg-color)', background: 'var(--comfy-menu-bg)',
    border: '1px solid var(--border-color)', borderRadius: '8px', padding: '24px', fontSize: '14px' });
  const title = element('h2', 'Export Turbo INT8 checkpoint', dialog);
  title.style.margin = '0 0 20px';
  title.style.fontSize = '18px';
  const form = element('form', '', dialog);
  const label = (text, control) => {
    const row = element('label', text, form);
    Object.assign(row.style, { display: 'grid', gap: '6px', marginBottom: '16px' });
    row.append(control);
    Object.assign(control.style, { font: 'inherit', padding: '8px', color: 'var(--input-text)',
      background: 'var(--comfy-input-bg)', border: '1px solid var(--border-color)', borderRadius: '4px' });
    return row;
  };
  const source = element('select');
  source.required = true;
  label('Source checkpoint', source);
  const name = element('input');
  name.required = true;
  name.value = 'krea2_turbo_custom_int8_convrot.safetensors';
  name.maxLength = 120;
  label('New checkpoint name', name);
  const stack = element('p', recipe.loras.length
    ? recipe.loras.map(x => `${x.name} · ${x.strength}`).join('\n') : 'No ordinary LoRAs selected.', form);
  Object.assign(stack.style, { whiteSpace: 'pre-line', overflowWrap: 'anywhere', margin: '0 0 16px' });
  element('p', 'Identity Edit and Style Reference remain runtime options.', form);
  const turbo = element('input');
  turbo.type = 'checkbox';
  turbo.required = true;
  const confirm = element('label', '', form);
  Object.assign(confirm.style, { display: 'flex', gap: '8px', alignItems: 'start', margin: '16px 0' });
  confirm.append(turbo);
  element('span', 'The source is already Turbo. This export does not distill RAW.', confirm);
  const status = element('p', 'Loading sources…', dialog);
  status.setAttribute('role', 'status');
  Object.assign(status.style, { overflowWrap: 'anywhere', minHeight: '20px', whiteSpace: 'pre-line' });
  const actions = element('div', '', dialog);
  Object.assign(actions.style, { display: 'flex', justifyContent: 'end', gap: '8px', marginTop: '20px' });
  const close = element('button', 'Close', actions);
  const cancel = element('button', 'Cancel export', actions);
  cancel.hidden = true;
  const submit = element('button', 'Export', actions);
  submit.disabled = true;
  submit.type = 'submit';
  submit.setAttribute('form', 'krea-export-form');
  form.id = 'krea-export-form';
  for (const button of [close, cancel, submit]) {
    button.className = 'comfy-menu-btn';
    Object.assign(button.style, { fontSize: '14px', padding: '8px 14px' });
  }
  let closed = false, currentJob, timer, initiated = false, finished = false;
  const finish = () => { closed = true; clearTimeout(timer); dialog.remove(); };
  close.onclick = finish;
  dialog.addEventListener('cancel', finish);
  document.body.append(dialog);
  dialog.showModal();
  const connected = () => app.graph?._nodes?.includes(node);
  const update = data => {
    if (closed) return;
    clearTimeout(timer);
    refreshChoices(app.graph?._nodes, data.checkpoints);
    app.canvas?.setDirty(true, true);
    currentJob = data.job;
    const active = isExportActive(currentJob);
    if (initiated && currentJob && !active) finished = true;
    for (const control of [source, name, turbo]) control.disabled = active;
    submit.disabled = active || finished || source.options.length === 0;
    cancel.hidden = !active;
    cancel.disabled = ['cancelling', 'cleaning'].includes(currentJob?.state);
    if (currentJob) {
      status.textContent = exportMessage(currentJob);
      if (currentJob.state === 'complete' && initiated && connected()) {
        initiated = false;
        status.textContent = useExport(node, currentJob.checkpoint, recipe)
          ? 'Checkpoint selected. Baked LoRAs removed from the runtime stack.'
          : 'Checkpoint ready. Your edited stack was preserved; select it below Size when ready.';
      }
    }
    if (active) timer = setTimeout(poll, 1500);
  };
  const poll = async () => {
    try { update(await request('export')); }
    catch (error) { if (!closed) status.textContent = error.message; }
  };
  cancel.onclick = async () => {
    cancel.disabled = true;
    try { update(await post('export-cancel', { id: currentJob.id })); }
    catch (error) { status.textContent = error.message; cancel.disabled = false; }
  };
  form.onsubmit = async event => {
    event.preventDefault();
    if (finished) return;
    submit.disabled = true;
    status.textContent = 'Validating export…';
    try {
      initiated = true;
      update(await post('export', { source: source.value, name: name.value.trim(), loras: recipe.loras, turbo: turbo.checked }));
    } catch (error) { initiated = false; status.textContent = error.message; submit.disabled = false; }
  };
  try {
    const data = await request('export-sources');
    if (closed) return;
    for (const entry of data.sources) {
      const option = element('option', entry.name + (entry.floating ? ' · floating' : ' · INT8') + ` · ${entry.location}`, source);
      option.value = entry.id;
      if (entry.name === recipe.checkpoint) option.selected = true;
    }
    status.textContent = data.sources.length ? '' : 'No compatible Turbo sources found.';
    submit.disabled = !data.sources.length;
    await poll();
  } catch (error) { if (!closed) status.textContent = error.message; }
}

app.registerExtension({
  name: 'mac-kernels.KreaCheckpointExport',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'BenchKreaWorkbench') return;
    const previous = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (...args) {
      const result = previous?.apply(this, args);
      args[1].push({ content: 'Export Turbo INT8 checkpoint…', callback: () => openExport(this) });
      return result;
    };
  },
});
