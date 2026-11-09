// Pinned ComfyUI frontend 1.53.10 production default, from defaultGraph.ts.
// Source map SHA256: 56c6ff4df75b833e01720bccc1b90343586d92f44f0dbfaeca30d33d7a754761
const stockGraph = {"last_node_id":71,"last_link_id":82,"nodes":[{"id":9,"type":"SaveImage","pos":[1279.9999726783708,319.9999392082668],"size":[300,420],"flags":{},"order":9,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":80}],"outputs":[],"properties":{"Node name for S&R":"SaveImage","cnr_id":"comfy-core","ver":"0.3.64","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":["ComfyUI"]},{"id":62,"type":"CLIPLoader","pos":[-239.9999987113997,420.0000536491848],"size":[340,169.3125],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"CLIP","type":"CLIP","links":[79,81]}],"properties":{"Node name for S&R":"CLIPLoader","cnr_id":"comfy-core","ver":"0.3.73","models":[{"name":"qwen_3_4b.safetensors","url":"https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors","directory":"text_encoders"}],"enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":["qwen_3_4b.safetensors","lumina2","default"]},{"id":63,"type":"VAELoader","pos":[659.9998200904802,699.9998629143215],"size":[320,106.65625],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[{"name":"VAE","type":"VAE","links":[74]}],"properties":{"Node name for S&R":"VAELoader","cnr_id":"comfy-core","ver":"0.3.73","models":[{"name":"ae.safetensors","url":"https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors","directory":"vae"}],"enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":["ae.safetensors"]},{"id":65,"type":"VAEDecode","pos":[1019.9998200904802,319.9999392082668],"size":[225,96],"flags":{},"order":8,"mode":0,"inputs":[{"name":"samples","type":"LATENT","link":73},{"name":"vae","type":"VAE","link":74}],"outputs":[{"name":"IMAGE","type":"IMAGE","slot_index":0,"links":[80]}],"properties":{"Node name for S&R":"VAEDecode","cnr_id":"comfy-core","ver":"0.3.64","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":[]},{"id":66,"type":"UNETLoader","pos":[-239.9999987113997,110.00000596546897],"size":[380,134.65625],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[{"name":"MODEL","type":"MODEL","links":[72]}],"properties":{"Node name for S&R":"UNETLoader","cnr_id":"comfy-core","ver":"0.3.73","models":[{"name":"z_image_turbo_bf16.safetensors","url":"https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors","directory":"diffusion_models"}],"enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":["z_image_turbo_bf16.safetensors","default"]},{"id":67,"type":"CLIPTextEncode","pos":[170.00001082534345,290.0000536491848],"size":[410,160],"flags":{},"order":4,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":79}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[76]}],"properties":{"Node name for S&R":"CLIPTextEncode","cnr_id":"comfy-core","ver":"0.3.73","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":["anime RPG game style, cute anime girl with gigantic fennec ears and a big fluffy fox tail with long wavy blonde hair and large blue eyes blonde colored eyelashes wearing a pink sweater a large oversized gold trimmed black winter coat and a long blue maxi skirt and a red scarf, she is sitting beside a campfire in the wilderness at night playing guitar with a milky way galaxy sky"]},{"id":68,"type":"EmptySD3LatentImage","pos":[310.0000489723161,700.0000155022121],"size":[260,168],"flags":{},"order":3,"mode":0,"inputs":[],"outputs":[{"name":"LATENT","type":"LATENT","slot_index":0,"links":[78]}],"properties":{"Node name for S&R":"EmptySD3LatentImage","cnr_id":"comfy-core","ver":"0.3.64","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":[1024,1024,1]},{"id":69,"type":"ModelSamplingAuraFlow","pos":[220.00001082534345,110.00000596546897],"size":[310,104],"flags":{},"order":6,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":72}],"outputs":[{"name":"MODEL","type":"MODEL","slot_index":0,"links":[75]}],"properties":{"Node name for S&R":"ModelSamplingAuraFlow","cnr_id":"comfy-core","ver":"0.3.64","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":[3]},{"id":70,"type":"KSampler","pos":[659.9998200904802,319.9999392082668],"size":[315,341.3125],"flags":{},"order":7,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":75},{"name":"positive","type":"CONDITIONING","link":76},{"name":"negative","type":"CONDITIONING","link":82},{"name":"latent_image","type":"LATENT","link":78}],"outputs":[{"name":"LATENT","type":"LATENT","slot_index":0,"links":[73]}],"properties":{"Node name for S&R":"KSampler","cnr_id":"comfy-core","ver":"0.3.64","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":[42,"fixed",8,1,"res_multistep","simple",1]},{"id":71,"type":"CLIPTextEncode","pos":[170.00001082534345,520.0000536491848],"size":[405.46875,140],"flags":{},"order":5,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":81}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[82]}],"properties":{"Node name for S&R":"CLIPTextEncode","cnr_id":"comfy-core","ver":"0.3.73","enableTabs":false,"tabWidth":65,"tabXOffset":10,"hasSecondTab":false,"secondTabText":"Send Back","secondTabOffset":80,"secondTabWidth":65},"widgets_values":["low quality, bad anatomy, extra digits, missing digits, extra limbs, missing limbs"]}],"links":[[72,66,0,69,0,"MODEL"],[73,70,0,65,0,"LATENT"],[74,63,0,65,1,"VAE"],[75,69,0,70,0,"MODEL"],[76,67,0,70,1,"CONDITIONING"],[78,68,0,70,3,"LATENT"],[79,62,0,67,0,"CLIP"],[80,65,0,9,0,"IMAGE"],[81,62,0,71,0,"CLIP"],[82,71,0,70,2,"CONDITIONING"]],"groups":[],"config":{},"extra":{"ds":{"scale":0.9,"offset":[416,110]}},"version":0.4};

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
  }
  return value;
}
function semanticGraph(graph) {
  const copy = structuredClone(graph);
  delete copy.id;
  for (const node of copy.nodes ?? []) {
    delete node.pos;
    delete node.size;
  }
  if (copy.extra) delete copy.extra.ds;
  return canonical(copy);
}
const stockSignature = JSON.stringify(semanticGraph(stockGraph));

export function shouldLoadWorkbenchDefault(graph) {
  if (!graph || typeof graph !== "object" || Array.isArray(graph)) return false;
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.links)) return false;
  // Unknown fields can carry authored work, including nested subgraphs.
  const allowed = new Set(["id", "last_node_id", "last_link_id", "nodes", "links", "groups", "config", "extra", "version"]);
  if (Object.keys(graph).some(key => !allowed.has(key))) return false;
  const extra = graph.extra ?? {};
  if (typeof extra !== "object" || extra === null || Array.isArray(extra)) return false;
  if (Object.keys(extra).some(key => key !== "ds")) return false;
  if (graph.nodes.length === 0 && graph.links.length === 0) {
    return Array.isArray(graph.groups) && graph.groups.length === 0
      && graph.config && typeof graph.config === "object" && !Array.isArray(graph.config)
      && Object.keys(graph.config).length === 0 && graph.version === 0.4
      && graph.last_node_id === 0 && graph.last_link_id === 0;
  }
  return JSON.stringify(semanticGraph(graph)) === stockSignature;
}

export function replaceWorkbenchDefault(graph, preset) {
  if (!shouldLoadWorkbenchDefault(graph)) return false;
  if (!preset || !Array.isArray(preset.nodes)
      || !preset.nodes.some(node => node.type === "BenchKreaWorkbench")) {
    throw new Error("Krea Workbench default workflow is unavailable");
  }
  const replacement = structuredClone(preset);
  if (Object.hasOwn(graph, "id")) replacement.id = graph.id;
  for (const key of Object.keys(graph)) delete graph[key];
  Object.assign(graph, replacement);
  return true;
}

// Saved graphs predate the optional size widget. Preserve their square output.
export function preserveLegacySize(graph, legacySize) {
  for (const node of graph?.nodes ?? []) {
    if (node.type !== "BenchKreaWorkbench" || !Array.isArray(node.widgets_values)) continue;
    const values = node.widgets_values;
    if (values.length === 14) values.push(false);
    if (values.length === 15) values.push(legacySize);
  }
}

// Run after size migration so earlier widgets retain their saved positions.
export function preserveLegacyCheckpoint(graph, defaultCheckpoint) {
  for (const node of graph?.nodes ?? []) {
    if (node.type !== "BenchKreaWorkbench" || !Array.isArray(node.widgets_values)) continue;
    if (node.widgets_values.length === 16) node.widgets_values.push(defaultCheckpoint);
  }
}
