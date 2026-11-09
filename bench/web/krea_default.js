import { app } from "/scripts/app.js";
import { replaceWorkbenchDefault, preserveLegacySize, preserveLegacyCheckpoint } from "./default_workflow.mjs";

let defaultWorkflow;
app.registerExtension({
  name: "mac-kernels.KreaWorkbenchDefault",
  async setup() {
    const response = await fetch("/krea-workbench/default-workflow");
    if (!response.ok) throw new Error("Could not load Krea Workbench default workflow");
    defaultWorkflow = await response.json();
  },
  beforeConfigureGraph(graphData) {
    replaceWorkbenchDefault(graphData, defaultWorkflow);
    preserveLegacySize(graphData, "Square 1:1 · 512 × 512");
    preserveLegacyCheckpoint(graphData, "krea2_turbo_int8_convrot.safetensors");
  },
});
