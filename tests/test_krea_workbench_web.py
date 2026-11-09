"""Real pinned frontend default fixture; execute the browser helper with Node."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from krea_workflows import ui_workflow

STOCK = json.loads('{"last_node_id": 71, "last_link_id": 82, "nodes": [{"id": 9, "type": "SaveImage", "pos": [1279.9999726783708, 319.9999392082668], "size": [300, 420], "flags": {}, "order": 9, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 80}], "outputs": [], "properties": {"Node name for S&R": "SaveImage", "cnr_id": "comfy-core", "ver": "0.3.64", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": ["ComfyUI"]}, {"id": 62, "type": "CLIPLoader", "pos": [-239.9999987113997, 420.0000536491848], "size": [340, 169.3125], "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [{"name": "CLIP", "type": "CLIP", "links": [79, 81]}], "properties": {"Node name for S&R": "CLIPLoader", "cnr_id": "comfy-core", "ver": "0.3.73", "models": [{"name": "qwen_3_4b.safetensors", "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors", "directory": "text_encoders"}], "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": ["qwen_3_4b.safetensors", "lumina2", "default"]}, {"id": 63, "type": "VAELoader", "pos": [659.9998200904802, 699.9998629143215], "size": [320, 106.65625], "flags": {}, "order": 1, "mode": 0, "inputs": [], "outputs": [{"name": "VAE", "type": "VAE", "links": [74]}], "properties": {"Node name for S&R": "VAELoader", "cnr_id": "comfy-core", "ver": "0.3.73", "models": [{"name": "ae.safetensors", "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors", "directory": "vae"}], "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": ["ae.safetensors"]}, {"id": 65, "type": "VAEDecode", "pos": [1019.9998200904802, 319.9999392082668], "size": [225, 96], "flags": {}, "order": 8, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 73}, {"name": "vae", "type": "VAE", "link": 74}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "slot_index": 0, "links": [80]}], "properties": {"Node name for S&R": "VAEDecode", "cnr_id": "comfy-core", "ver": "0.3.64", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": []}, {"id": 66, "type": "UNETLoader", "pos": [-239.9999987113997, 110.00000596546897], "size": [380, 134.65625], "flags": {}, "order": 2, "mode": 0, "inputs": [], "outputs": [{"name": "MODEL", "type": "MODEL", "links": [72]}], "properties": {"Node name for S&R": "UNETLoader", "cnr_id": "comfy-core", "ver": "0.3.73", "models": [{"name": "z_image_turbo_bf16.safetensors", "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors", "directory": "diffusion_models"}], "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": ["z_image_turbo_bf16.safetensors", "default"]}, {"id": 67, "type": "CLIPTextEncode", "pos": [170.00001082534345, 290.0000536491848], "size": [410, 160], "flags": {}, "order": 4, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 79}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [76]}], "properties": {"Node name for S&R": "CLIPTextEncode", "cnr_id": "comfy-core", "ver": "0.3.73", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": ["anime RPG game style, cute anime girl with gigantic fennec ears and a big fluffy fox tail with long wavy blonde hair and large blue eyes blonde colored eyelashes wearing a pink sweater a large oversized gold trimmed black winter coat and a long blue maxi skirt and a red scarf, she is sitting beside a campfire in the wilderness at night playing guitar with a milky way galaxy sky"]}, {"id": 68, "type": "EmptySD3LatentImage", "pos": [310.0000489723161, 700.0000155022121], "size": [260, 168], "flags": {}, "order": 3, "mode": 0, "inputs": [], "outputs": [{"name": "LATENT", "type": "LATENT", "slot_index": 0, "links": [78]}], "properties": {"Node name for S&R": "EmptySD3LatentImage", "cnr_id": "comfy-core", "ver": "0.3.64", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": [1024, 1024, 1]}, {"id": 69, "type": "ModelSamplingAuraFlow", "pos": [220.00001082534345, 110.00000596546897], "size": [310, 104], "flags": {}, "order": 6, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 72}], "outputs": [{"name": "MODEL", "type": "MODEL", "slot_index": 0, "links": [75]}], "properties": {"Node name for S&R": "ModelSamplingAuraFlow", "cnr_id": "comfy-core", "ver": "0.3.64", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": [3]}, {"id": 70, "type": "KSampler", "pos": [659.9998200904802, 319.9999392082668], "size": [315, 341.3125], "flags": {}, "order": 7, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 75}, {"name": "positive", "type": "CONDITIONING", "link": 76}, {"name": "negative", "type": "CONDITIONING", "link": 82}, {"name": "latent_image", "type": "LATENT", "link": 78}], "outputs": [{"name": "LATENT", "type": "LATENT", "slot_index": 0, "links": [73]}], "properties": {"Node name for S&R": "KSampler", "cnr_id": "comfy-core", "ver": "0.3.64", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": [42, "fixed", 8, 1, "res_multistep", "simple", 1]}, {"id": 71, "type": "CLIPTextEncode", "pos": [170.00001082534345, 520.0000536491848], "size": [405.46875, 140], "flags": {}, "order": 5, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 81}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [82]}], "properties": {"Node name for S&R": "CLIPTextEncode", "cnr_id": "comfy-core", "ver": "0.3.73", "enableTabs": false, "tabWidth": 65, "tabXOffset": 10, "hasSecondTab": false, "secondTabText": "Send Back", "secondTabOffset": 80, "secondTabWidth": 65}, "widgets_values": ["low quality, bad anatomy, extra digits, missing digits, extra limbs, missing limbs"]}], "links": [[72, 66, 0, 69, 0, "MODEL"], [73, 70, 0, 65, 0, "LATENT"], [74, 63, 0, 65, 1, "VAE"], [75, 69, 0, 70, 0, "MODEL"], [76, 67, 0, 70, 1, "CONDITIONING"], [78, 68, 0, 70, 3, "LATENT"], [79, 62, 0, 67, 0, "CLIP"], [80, 65, 0, 9, 0, "IMAGE"], [81, 62, 0, 71, 0, "CLIP"], [82, 71, 0, 70, 2, "CONDITIONING"]], "groups": [], "config": {}, "extra": {"ds": {"scale": 0.9, "offset": [416, 110]}}, "version": 0.4}')


def test_default_workflow_preserves_authored_graphs_and_uses_fresh_clones():
    node = shutil.which("node")
    assert node, "Node 24 is required for the frontend helper proof"
    module = (Path(__file__).resolve().parents[1] / "bench/web/default_workflow.mjs").as_uri()
    source = r'''
import assert from "node:assert/strict";
import { shouldLoadWorkbenchDefault, replaceWorkbenchDefault } from MODULE;
const stock = STOCK;
const blank = {last_node_id:0,last_link_id:0,nodes:[],links:[],groups:[],config:{},extra:{},version:0.4};
const preset = PRESET;
const originalPrompt = preset.nodes.find(n=>n.type==="BenchKreaWorkbench").widgets_values[0];
assert.equal(shouldLoadWorkbenchDefault(stock),true);
assert.equal(shouldLoadWorkbenchDefault(blank),true);
const viewed = structuredClone(stock);
viewed.id = "draft-id";
viewed.extra.ds = {scale:1.5,offset:[50,80]};
viewed.nodes[0].pos = [99,99];
assert.equal(shouldLoadWorkbenchDefault(viewed),true);
assert.equal(replaceWorkbenchDefault(viewed,preset),true);
assert.equal(viewed.id,"draft-id");
assert.equal(replaceWorkbenchDefault(viewed,preset),false);
viewed.nodes[0].widgets_values[0] = "edited";
assert.equal(preset.nodes[0].widgets_values[0],originalPrompt);
const second = structuredClone(blank);
replaceWorkbenchDefault(second,preset);
assert.equal(second.nodes[0].widgets_values[0],originalPrompt);
for (const mutate of [
 g => g.nodes.find(n=>n.type==="UNETLoader").widgets_values[0]="krea2_turbo_int8_convrot.safetensors",
 g => g.nodes.find(n=>n.id===67).widgets_values[0]="my own prompt",
 g => g.links[0][1]=999,
 g => g.nodes.push({id:999,type:"BenchKreaWorkbench"}),
 g => g.extra.note="authored metadata",
 g => g.definitions={subgraphs:[{id:"nested"}]},
 g => g.groups.push({title:"my group"}),
]) {
 const changed=structuredClone(stock); mutate(changed);
 const before=structuredClone(changed);
 assert.equal(replaceWorkbenchDefault(changed,preset),false);
 assert.deepEqual(changed,before);
}
for(const candidate of [null,{},[], {...blank,definitions:{}},{...blank,extra:{custom:"data"}},{...blank,groups:[{}]}]) {
 assert.equal(shouldLoadWorkbenchDefault(candidate),false);
}
assert.throws(()=>replaceWorkbenchDefault(structuredClone(blank),{}),/unavailable/);
console.log("stock, blank, draft identity, clone isolation, custom and subgraph preservation passed");
'''
    source = source.replace("MODULE", json.dumps(module)).replace("STOCK", json.dumps(STOCK))
    source = source.replace("PRESET", json.dumps(ui_workflow("text")))
    result = subprocess.run([node, "--input-type=module", "-e", source], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_registered_extension_waits_for_default_fetch_before_configuring():
    node = shutil.which("node")
    assert node, "Node 24 is required for the frontend extension proof"
    web = Path(__file__).resolve().parents[1] / "bench/web"
    extension = (web / "krea_default.js").read_text()
    # Browser app registration/fetch are external boundaries; execute actual hooks.
    extension = extension.replace('import { app } from "/scripts/app.js";', 'const app = globalThis.testApp;')
    extension = extension.replace('"./default_workflow.mjs"', json.dumps((web / "default_workflow.mjs").as_uri()))
    source = r'''
import assert from "node:assert/strict";
let registered;
globalThis.testApp = {registerExtension(extension){registered=extension;}};
let resolveFetch;
globalThis.fetch = async path => {
 assert.equal(path,"/krea-workbench/default-workflow");
 return await new Promise(resolve=>{resolveFetch=resolve;});
};
await import("data:text/javascript,"+encodeURIComponent(EXTENSION));
assert.equal(registered.name,"mac-kernels.KreaWorkbenchDefault");
const setup = registered.setup();
resolveFetch({ok:true,json:async()=>(PRESET)});
await setup;
const graph = STOCK;
registered.beforeConfigureGraph(graph);
assert.equal(graph.nodes[0].type,"BenchKreaWorkbench");
graph.nodes[0].widgets_values[0]="saved Krea draft";
registered.beforeConfigureGraph(graph);
assert.equal(graph.nodes[0].widgets_values[0],"saved Krea draft");
for(const count of [14,15,16,17]) {
 const saved=structuredClone(graph);
 const values=saved.nodes[0].widgets_values;
 values.length=count;
 if(count>=15)values[14]=true;
 if(count>=16)values[15]="Landscape 3:2 · 1248 × 832";
 if(count>=17)values[16]="custom-turbo.safetensors";
 const before=structuredClone(values);
 registered.beforeConfigureGraph(saved);
 assert.deepEqual(values.slice(0,count),before);
 assert.equal(values.length,17);
 assert.equal(values[14],count===14?false:true);
 assert.equal(values[15],count<16?"Square 1:1 · 512 × 512":"Landscape 3:2 · 1248 × 832");
 assert.equal(values[16],count<17?"krea2_turbo_int8_convrot.safetensors":"custom-turbo.safetensors");
 const migrated=structuredClone(saved);
 registered.beforeConfigureGraph(saved);
 assert.deepEqual(saved,migrated);
}
globalThis.fetch = async()=>({ok:false});
await assert.rejects(registered.setup(),/Could not load/);
'''
    source = source.replace("EXTENSION", json.dumps(extension)).replace("PRESET", json.dumps(ui_workflow("text"))).replace("STOCK", json.dumps(STOCK))
    result = subprocess.run([node, "--input-type=module", "-e", source], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_saved_workbench_sizes_migrate_without_resetting_prompt_or_spectral():
    node = shutil.which("node")
    assert node
    module = (Path(__file__).resolve().parents[1] / "bench/web/default_workflow.mjs").as_uri()
    source = r'''
import assert from "node:assert/strict";
import {preserveLegacySize} from MODULE;
const legacySize = "Square 1:1 · 512 × 512";
for (const count of [14,15,16,17]) {
 const values=Array.from({length:count},(_,i)=> i===0 ? "my prompt" : i);
 if(count>=15)values[14]=true;
 const before=structuredClone(values);
 const graph={nodes:[{type:"BenchKreaWorkbench",widgets_values:values}]};
 preserveLegacySize(graph,legacySize);
 assert.deepEqual(values.slice(0,count),before);
 if(count===14)assert.deepEqual(values.slice(14),[false,legacySize]);
 if(count===15)assert.equal(values[15],legacySize);
 if(count>=16)assert.deepEqual(values,before);
 const after=structuredClone(graph);
 preserveLegacySize(graph,legacySize);
 assert.deepEqual(graph,after);
}
const unrelated={nodes:[{type:"OtherNode",widgets_values:Array(14).fill("x")}]};
const before=structuredClone(unrelated);
preserveLegacySize(unrelated,legacySize);
assert.deepEqual(unrelated,before);
'''
    result=subprocess.run([node,"--input-type=module","-e",source.replace("MODULE",json.dumps(module))],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_browser_legacy_size_matches_authoritative_preset():
    from krea_sizes import LEGACY_SIZE
    extension = (Path(__file__).resolve().parents[1] / "bench/web/krea_default.js").read_text()
    assert f'preserveLegacySize(graphData, "{LEGACY_SIZE}")' in extension


def test_browser_default_checkpoint_matches_backend_and_workflows():
    from krea_checkpoints import DEFAULT_CHECKPOINT
    from krea_server import UNET
    from krea_workflows import DEFAULT_CHECKPOINT as WORKFLOW_DEFAULT
    assert UNET == WORKFLOW_DEFAULT == DEFAULT_CHECKPOINT
    extension = (Path(__file__).resolve().parents[1] / "bench/web/krea_default.js").read_text()
    assert f'preserveLegacyCheckpoint(graphData, "{DEFAULT_CHECKPOINT}")' in extension


def test_browser_legacy_checkpoint_matches_api_default():
    from krea_workflows import DEFAULT_CHECKPOINT
    extension = (Path(__file__).resolve().parents[1] / "bench/web/krea_default.js").read_text()
    assert f'preserveLegacyCheckpoint(graphData, "{DEFAULT_CHECKPOINT}")' in extension
