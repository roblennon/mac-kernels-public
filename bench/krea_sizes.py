"""Named, patch-aligned output sizes for the dedicated Krea Workbench."""
# Krea2-Harness's published 1K anchors; 3:4 mirrors its 4:3 anchor.
# https://github.com/ANe5s/ComfyUI-Krea2-Harness#krea2-resolution-selector
SIZE_PRESETS = {
    'Portrait 2:3 · 832 × 1248': (832, 1248),
    'Landscape 3:2 · 1248 × 832': (1248, 832),
    'Square 1:1 · 1024 × 1024': (1024, 1024),
    'Portrait 3:4 · 896 × 1184': (896, 1184),
    'Landscape 4:3 · 1184 × 896': (1184, 896),
    'Portrait 4:5 · 928 × 1152': (928, 1152),
    'Tall 9:16 · 768 × 1376': (768, 1376),
    'Wide 16:9 · 1376 × 768': (1376, 768),
    'Cinema 2.35:1 · 1568 × 672': (1568, 672),
    'Square 1:1 · 512 × 512': (512, 512),
}
DEFAULT_SIZE = 'Portrait 2:3 · 832 × 1248'

LEGACY_SIZE = 'Square 1:1 · 512 × 512'


def output_size(preset):
    if not isinstance(preset, str) or preset not in SIZE_PRESETS:
        raise ValueError('Choose a supported Krea output-size preset')
    return SIZE_PRESETS[preset]
