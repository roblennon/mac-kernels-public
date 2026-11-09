"""Real CPU quantization, merge arithmetic, format loading and atomic failures."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_streaming_export_real_math_and_admission():
    script = r'''
import json
import tempfile
from pathlib import Path
from bench import krea_export as export
torch, utils, quantize, dequantize = export._runtime()
from safetensors import safe_open
from safetensors.torch import save_file, load_file
import comfy.ops as ops
from comfy.quant_ops import QuantizedTensor

generator = torch.Generator().manual_seed(719)
layers = ('attn.gate','attn.wk','attn.wo','attn.wq','attn.wv','mlp.down','mlp.gate','mlp.up')
weights = {f'blocks.{i}.{layer}.weight': torch.randn(2,256,generator=generator)*.05
           for i in range(28) for layer in layers}
global_key = 'last.linear.weight'
weights[global_key] = torch.randn(2,256,generator=generator)*.05
weights['last.norm.scale'] = torch.ones(2)
original = {}
for key, weight in weights.items():
    if key.startswith('blocks.'):
        q,s = quantize(weight,256)
        original[key] = q
        original[key[:-7]+'.weight_scale'] = s
        original[key[:-7]+'.comfy_quant'] = torch.tensor(list(json.dumps(export.QUANT_CONFIG).encode()),dtype=torch.uint8)
    else:
        original[key] = weight.bfloat16() if key.endswith('.weight') else weight

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    template = directory/'template.safetensors'
    source = directory/'float.safetensors'
    save_file(original,str(template))
    # Reduced dimensions retain the real 224-layer architecture/count and layout.
    # Only the production header pin is substituted, not the format/shape checks.
    export.TEMPLATE_HEADER_SHA256 = export.inspect_header(template)['header_sha256']
    save_file({k:v.half() for k,v in weights.items()},str(source))
    result_path = directory/'float-export.safetensors'
    result = export.export_checkpoint(source,result_path,source_variant='turbo',reference=template,row_chunk=1)
    state = load_file(str(result_path))
    assert not result['provenance']['source']['quantized']
    assert set(state)==set(original)
    for key, weight in weights.items():
        if key.startswith('blocks.'):
            q,s=quantize(weight.half().float(),256)
            assert torch.equal(state[key],q)
            assert torch.allclose(state[key[:-7]+'.weight_scale'],s,atol=1e-9)
        else:
            assert state[key].dtype==original[key].dtype
    # The actual Comfy quantized Linear loader accepts the exported wire format.
    key = 'blocks.0.attn.wq.weight'
    prefix=key[:-7]+'.'
    operations=ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    linear=operations.Linear(256,2,bias=False,device=torch.device('cpu'))
    linear.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    assert isinstance(linear.weight,QuantizedTensor)
    assert linear.weight._params.convrot and linear.weight._params.convrot_groupsize==256

    def expect_failure(call, text):
        try:
            call()
        except (ValueError,RuntimeError,FileExistsError) as error:
            assert text in str(error),(text,str(error))
        else:
            raise AssertionError('Expected rejection: '+text)

    # Untouched INT8 payloads are byte-identical, despite new header provenance.
    copied=directory/'copy.safetensors'
    copy_receipt=export.export_checkpoint(template,copied,source_variant='turbo',reference=template,row_chunk=1)
    assert copy_receipt['provenance']['source']['quantized']
    with safe_open(str(copied),framework='pt',device='cpu') as reader:
        metadata=json.loads(reader.metadata()[export.METADATA_KEY])
        assert metadata['architecture']=='krea2' and metadata['base_variant']=='turbo'
    assert all(torch.equal(v,load_file(str(copied))[k]) for k,v in original.items())
    expect_failure(lambda: export.export_checkpoint(template,copied,source_variant='turbo',reference=template),'new .safetensors')
    expect_failure(lambda: export.export_checkpoint(template,directory/'raw.safetensors',source_variant='raw',reference=template),'no RAW distillation')

    rank=3
    up=torch.randn(2,rank,generator=generator)*.1
    down=torch.randn(rank,256,generator=generator)*.1
    alpha=6.
    # Use the same diffusers names as our real RAW trainer exports.
    factors={'transformer.transformer_blocks.0.attn.to_q.lora_A.weight':down,
             'transformer.transformer_blocks.0.attn.to_q.lora_B.weight':up,
             'transformer.transformer_blocks.0.attn.to_q.alpha':torch.tensor(alpha),
             'diffusion_model.last.linear.lora_down.weight':down.clone(),
             'diffusion_model.last.linear.lora_up.weight':up.clone()}
    lora=directory/'style.safetensors'
    save_file(factors,str(lora))
    merged=directory/'merged.safetensors'
    recipe=[{'path':lora,'strength':-.75}]
    merged_receipt=export.export_checkpoint(template,merged,source_variant='turbo',reference=template,loras=recipe,row_chunk=1)
    output=load_file(str(merged))
    dense=dequantize(original[key],original[key[:-7]+'.weight_scale'],256)-.75*(alpha/rank)*(up@down)
    q,s=quantize(dense,256)
    assert torch.equal(q,output[key])
    assert torch.allclose(s,output[key[:-7]+'.weight_scale'],atol=1e-9)
    assert torch.equal(output[global_key],(original[global_key].float()-.75*(up@down)).bfloat16())
    untouched='blocks.1.attn.wq.weight'
    assert torch.equal(output[untouched],original[untouched])
    assert merged_receipt['provenance']['baked_loras'][0]['name']=='style.safetensors'
    rebaked=directory/'rebake.safetensors'
    chained=export.export_checkpoint(merged,rebaked,source_variant='turbo',reference=template)
    assert chained['provenance']['baked_loras']==merged_receipt['provenance']['baked_loras']
    expect_failure(lambda:export.export_checkpoint(merged,directory/'replay.safetensors',source_variant='turbo',reference=template,loras=recipe),'previously baked')

    # Sum two independently scaled adapters before quantization, not their codes.
    second=directory/'second.safetensors'
    save_file({k:v*2 if '.lora_' in k else v for k,v in factors.items()},str(second))
    stacked=directory/'stacked.safetensors'
    export.export_checkpoint(template,stacked,source_variant='turbo',reference=template,
        loras=[*recipe,{'path':second,'strength':.5}],row_chunk=2)
    dense=dequantize(original[key],original[key[:-7]+'.weight_scale'],256)+(-.75+.5*4)*(alpha/rank)*(up@down)
    assert torch.equal(load_file(str(stacked))[key],quantize(dense,256)[0])

    invalid=directory/'bad.safetensors'
    missing=directory/'failed.safetensors'
    save_file({**factors,'orphan.alpha':torch.tensor(1.)},str(invalid))
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template,
        loras=[{'path':invalid,'strength':1.}]),'unconsumed')
    assert not missing.exists()
    save_file({**factors,'transformer.transformer_blocks.0.attn.to_q.lora_A.weight':down*float('nan')},str(invalid))
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template,
        loras=[{'path':invalid,'strength':1.}]),'must be finite')
    assert not missing.exists()
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template,
        loras=recipe*2),'Duplicate')
    special=directory/'krea2_style_reference.safetensors'
    save_file(factors,str(special))
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template,
        loras=[{'path':special,'strength':1.}]),'Special reference')
    export.SPECIAL_HASHES.add(export._hash(export.inspect_header(lora)))
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template,loras=recipe),'Special reference')

    # Preexisting recipe fingerprints survive a rename, and the total stack is bounded.
    export.SPECIAL_HASHES.discard(export._hash(export.inspect_header(lora)))
    renamed=directory/'renamed.safetensors'
    renamed.write_bytes(lora.read_bytes())
    expect_failure(lambda:export.export_checkpoint(merged,missing,source_variant='turbo',reference=template,
        loras=[{'path':renamed,'strength':1.}]),'previously baked')
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template,
        loras=[{'path':lora,'strength':2.01}]),'[-2, 2]')
    five=[]
    for i in range(5):
        path=directory/f'adapter-{i}.safetensors'
        save_file({k:v*(i+2) for k,v in factors.items()},str(path))
        five.append({'path':path,'strength':.1})
    five_path=directory/'five.safetensors'
    export.export_checkpoint(template,five_path,source_variant='turbo',reference=template,loras=five)
    expect_failure(lambda:export.export_checkpoint(five_path,missing,source_variant='turbo',reference=template,loras=recipe),'five total')
    nonfinite=directory/'nonfinite-base.safetensors'
    save_file({**original,global_key:original[global_key]*float('inf')},str(nonfinite))
    expect_failure(lambda:export.export_checkpoint(nonfinite,missing,source_variant='turbo',reference=template),'Nonfinite checkpoint')
    assert not missing.exists()
    negative=directory/'negative-scales.safetensors'
    scale_key=key[:-7]+'.weight_scale'
    save_file({**original,scale_key:-original[scale_key]},str(negative))
    expect_failure(lambda:export.export_checkpoint(negative,missing,source_variant='turbo',reference=template),'scales must be positive')
    assert not missing.exists()

    # A concurrent publisher keeps its own bytes; our hardlink cannot overwrite it.
    original_link=export.os.link
    def concurrent_link(candidate, destination):
        Path(destination).write_bytes(b'other publisher')
        return original_link(candidate,destination)
    export.os.link=concurrent_link
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template),'File exists')
    assert missing.read_bytes()==b'other publisher'
    missing.unlink()
    export.os.link=original_link

    # Reading many unique payloads must not retain file mappings in process RSS.
    import gc
    import resource
    import struct
    import sys
    medium=directory/'bounded-read.safetensors'
    count=64
    payload_bytes=2*2**20
    header={f'matrix.{i}':{'dtype':'F32','shape':[1024,512],
            'data_offsets':[i*payload_bytes,(i+1)*payload_bytes]} for i in range(count)}
    raw=json.dumps(header,separators=(',',':')).encode()
    raw+=b' '*(-len(raw)%8)
    with medium.open('wb') as f:
        f.write(struct.pack('<Q',len(raw)))
        f.write(raw)
        chunk=b'\0'*payload_bytes
        for _ in range(count):f.write(chunk)
    del chunk
    asset=export.inspect_header(medium)
    gc.collect()
    factor=1 if sys.platform=='darwin' else 1024
    before_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor
    with medium.open('rb') as stream:
        for i in range(count):
            value=export._read_tensor(asset,stream,f'matrix.{i}',torch)
            assert value.shape==(1024,512) and value.sum().item()==0
    del value
    after_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor
    assert after_peak-before_peak<32*2**20,(before_peak,after_peak)

    # SIGTERM follows normal exception cleanup before any checkpoint publication.
    export.sys.argv=['krea_export','--source',str(template),'--output',str(missing),'--source-variant','turbo']
    real_export=export.export_checkpoint
    def stopped_export(*args,**kwargs):
        import os
        import signal
        with tempfile.TemporaryDirectory(prefix='.krea-export-',dir=directory):
            os.kill(os.getpid(),signal.SIGTERM)
    export.export_checkpoint=stopped_export
    try:
        export.main()
    except SystemExit as error:
        assert error.code==130
    else:
        raise AssertionError('SIGTERM did not interrupt export')
    export.export_checkpoint=real_export
    assert not missing.exists() and not list(directory.glob('.krea-export-*'))

    # Source replacement during publication fails closed and cleans temp payloads.
    original_hash=export._hash
    def mutate_after_hash(asset):
        digest=original_hash(asset)
        if asset['path'].endswith('complete.safetensors'):
            with template.open('ab') as f:f.write(b'x')
        return digest
    export._hash=mutate_after_hash
    expect_failure(lambda:export.export_checkpoint(template,missing,source_variant='turbo',reference=template),'input changed')
    assert not missing.exists()
    assert not list(directory.glob('.krea-export-*'))
print('streaming export arithmetic, loader format, admission and atomic failure checks passed')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
