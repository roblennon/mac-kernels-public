/** Frozen recipes refer to ordinary LoRA controls; reference modes stay runtime. */
export function isExportActive(job) {
  return ['preparing', 'exporting', 'installing', 'cancelling', 'cleaning'].includes(job?.state);
}

export function exportMessage(job) {
  return [job?.error, job?.progress].filter(Boolean).join('\n');
}

export function captureRecipe(node) {
  const widgets = Object.fromEntries((node.widgets ?? []).map(w => [w.name, w.value]));
  const loras = [];
  for (let i = 1; i <= 5; i++) {
    const name = widgets[`lora_${i}`];
    const strength = widgets[`strength_${i}`];
    if (name && strength !== 0) loras.push({ name, strength });
  }
  return { checkpoint: widgets.checkpoint, loras };
}

export function refreshChoices(nodes, names) {
  for (const node of nodes ?? []) {
    if (node.comfyClass !== 'BenchKreaWorkbench' && node.type !== 'BenchKreaWorkbench') continue;
    const widget = node.widgets?.find(w => w.name === 'checkpoint');
    if (widget) widget.options.values = [...names];
  }
}

export function useExport(node, checkpoint, recipe) {
  if (JSON.stringify(captureRecipe(node)) !== JSON.stringify(recipe)) return false;
  node.graph?.beforeChange();
  for (const widget of node.widgets ?? []) {
    if (/^lora_[1-5]$/.test(widget.name)
      && node.widgets.find(w => w.name === `strength_${widget.name.slice(-1)}`)?.value !== 0
      && recipe.loras.some(x => x.name === widget.value)) widget.value = '';
    if (widget.name === 'checkpoint') widget.value = checkpoint;
  }
  node.graph?.afterChange();
  node.setDirtyCanvas?.(true, true);
  return true;
}
