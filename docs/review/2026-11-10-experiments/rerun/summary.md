## Decode step: ratio of times within a process (above 1 means the second arm is faster)

### B1_P32

- eager_dynamic/eager_static: 0.937 (range 0.924–0.991, n=4)
- eager_static/compiled: 1.605 (range 1.587–1.797, n=4)
- eager_dynamic/compiled: 1.554 (range 1.475–1.660, n=4)
- compiled/kernel: 1.010 (range 0.914–1.047, n=4)
- eager_static/kernel: 1.643 (range 1.582–1.680, n=4)
- eager_dynamic/kernel: 1.553 (range 1.470–1.629, n=4)
- A/A compiled_a/compiled_b: 1.041 (range 1.012–1.092, n=4)
- A/A kernel_a/kernel_b: 0.982 (range 0.964–1.073, n=4)

### B1_P512

- eager_dynamic/eager_static: 0.971 (range 0.942–1.068, n=4)
- eager_static/compiled: 1.719 (range 1.643–1.852, n=4)
- eager_dynamic/compiled: 1.722 (range 1.621–1.777, n=4)
- compiled/kernel: 1.016 (range 0.990–1.068, n=4)
- eager_static/kernel: 1.781 (range 1.686–1.835, n=4)
- eager_dynamic/kernel: 1.743 (range 1.702–1.801, n=4)
- A/A compiled_a/compiled_b: 0.923 (range 0.856–1.092, n=4)
- A/A kernel_a/kernel_b: 0.964 (range 0.936–0.976, n=4)

### B4_P32

- eager_dynamic/eager_static: 0.972 (range 0.924–1.068, n=4)
- eager_static/compiled: 1.224 (range 1.065–1.265, n=4)
- eager_dynamic/compiled: 1.144 (range 1.118–1.283, n=4)
- compiled/kernel: 1.416 (range 1.272–1.436, n=4)
- eager_static/kernel: 1.650 (range 1.529–1.773, n=4)
- eager_dynamic/kernel: 1.632 (range 1.574–1.639, n=4)
- A/A compiled_a/compiled_b: 0.968 (range 0.823–1.196, n=4)
- A/A kernel_a/kernel_b: 1.006 (range 0.988–1.032, n=4)

### B8_P32

- eager_dynamic/eager_static: 0.996 (range 0.993–1.005, n=4)
- eager_static/compiled: 1.185 (range 1.141–1.199, n=4)
- eager_dynamic/compiled: 1.185 (range 1.133–1.196, n=4)
- compiled/kernel: 1.658 (range 1.592–1.703, n=4)
- eager_static/kernel: 1.931 (range 1.908–2.009, n=4)
- eager_dynamic/kernel: 1.920 (range 1.904–2.019, n=4)
- A/A compiled_a/compiled_b: 0.973 (range 0.861–0.991, n=4)
- A/A kernel_a/kernel_b: 1.005 (range 0.991–1.013, n=4)

### B16_P32

- eager_dynamic/eager_static: 1.005 (range 0.988–1.033, n=4)
- eager_static/compiled: 1.155 (range 1.138–1.175, n=4)
- eager_dynamic/compiled: 1.171 (range 1.124–1.193, n=4)
- compiled/kernel: 1.765 (range 1.693–2.020, n=4)
- eager_static/kernel: 2.039 (range 1.990–2.298, n=4)
- eager_dynamic/kernel: 2.067 (range 2.021–2.271, n=4)
- A/A compiled_a/compiled_b: 0.993 (range 0.959–1.010, n=4)
- A/A kernel_a/kernel_b: 0.997 (range 0.925–1.033, n=4)

### B16_P512

- eager_dynamic/eager_static: 1.084 (range 1.074–1.087, n=4)
- eager_static/compiled: 1.118 (range 1.113–1.149, n=4)
- eager_dynamic/compiled: 1.214 (range 1.196–1.243, n=4)
- compiled/kernel: 1.692 (range 1.528–1.776, n=4)
- eager_static/kernel: 1.891 (range 1.755–1.979, n=4)
- eager_dynamic/kernel: 2.054 (range 1.900–2.125, n=4)
- A/A compiled_a/compiled_b: 1.003 (range 0.998–1.006, n=4)
- A/A kernel_a/kernel_b: 1.051 (range 1.036–1.078, n=4)

## Prompt lookup: plain_a time over arm time (above 1 means the arm is faster)

### code_edit

- decode_s plain_a/spec: 2.078 (range 2.016–2.521, n=4)
- whole_s plain_a/spec: 2.010 (range 1.957–2.425, n=4)
- decode_s_per_token plain_a/spec: 2.078 (range 2.016–2.521, n=4)
- decode_s plain_a/aspec: 2.066 (range 2.008–2.516, n=4)
- whole_s plain_a/aspec: 2.000 (range 1.948–2.421, n=4)
- decode_s_per_token plain_a/aspec: 2.066 (range 2.008–2.516, n=4)
- decode_s plain_a/plain_b: 1.012 (range 0.969–1.254, n=4)
- whole_s plain_a/plain_b: 1.012 (range 0.970–1.246, n=4)
- decode_s_per_token plain_a/plain_b: 1.012 (range 0.969–1.254, n=4)
- decode_s plain_a/spec_kernel: 2.920 (range 2.807–3.532, n=4)
- whole_s plain_a/spec_kernel: 2.758 (range 2.656–3.312, n=4)
- decode_s_per_token plain_a/spec_kernel: 2.920 (range 2.807–3.532, n=4)
- decode_s plain_a/aspec_kernel: 2.973 (range 2.849–3.515, n=4)
- whole_s plain_a/aspec_kernel: 2.800 (range 2.690–3.302, n=4)
- decode_s_per_token plain_a/aspec_kernel: 2.973 (range 2.849–3.515, n=4)
- first divergence from plain_a (None = identical): {'plain_a': [None, None, None, None], 'spec': [None, None, None, None], 'aspec': [None, None, None, None], 'plain_b': [None, None, None, None], 'spec_kernel': [None, None, None, None], 'aspec_kernel': [None, None, None, None]}

### doc_edit

- decode_s plain_a/spec: 2.079 (range 2.048–2.154, n=4)
- whole_s plain_a/spec: 1.996 (range 1.954–2.052, n=4)
- decode_s_per_token plain_a/spec: 2.079 (range 2.048–2.154, n=4)
- decode_s plain_a/aspec: 2.222 (range 2.148–2.267, n=4)
- whole_s plain_a/aspec: 2.107 (range 2.041–2.161, n=4)
- decode_s_per_token plain_a/aspec: 2.222 (range 2.148–2.267, n=4)
- decode_s plain_a/plain_b: 1.012 (range 0.981–1.021, n=4)
- whole_s plain_a/plain_b: 1.013 (range 0.983–1.020, n=4)
- decode_s_per_token plain_a/plain_b: 1.012 (range 0.981–1.021, n=4)
- decode_s plain_a/spec_kernel: 3.138 (range 2.935–3.230, n=4)
- whole_s plain_a/spec_kernel: 2.855 (range 2.696–2.934, n=4)
- decode_s_per_token plain_a/spec_kernel: 3.138 (range 2.935–3.230, n=4)
- decode_s plain_a/aspec_kernel: 3.309 (range 3.257–3.423, n=4)
- whole_s plain_a/aspec_kernel: 3.008 (range 2.955–3.071, n=4)
- decode_s_per_token plain_a/aspec_kernel: 3.309 (range 3.257–3.423, n=4)
- first divergence from plain_a (None = identical): {'plain_a': [None, None, None, None], 'spec': [None, None, None, None], 'aspec': [None, None, None, None], 'plain_b': [None, None, None, None], 'spec_kernel': [None, None, None, None], 'aspec_kernel': [None, None, None, None]}

### free_writing

- decode_s plain_a/spec: 0.579 (range 0.567–0.599, n=4)
- whole_s plain_a/spec: 0.583 (range 0.571–0.602, n=4)
- decode_s_per_token plain_a/spec: 0.579 (range 0.567–0.599, n=4)
- decode_s plain_a/aspec: 0.917 (range 0.867–1.006, n=4)
- whole_s plain_a/aspec: 0.918 (range 0.867–1.007, n=4)
- decode_s_per_token plain_a/aspec: 0.917 (range 0.867–1.006, n=4)
- decode_s plain_a/plain_b: 1.009 (range 0.970–1.110, n=4)
- whole_s plain_a/plain_b: 1.009 (range 0.971–1.110, n=4)
- decode_s_per_token plain_a/plain_b: 1.009 (range 0.970–1.110, n=4)
- decode_s plain_a/spec_kernel: 0.819 (range 0.778–0.905, n=4)
- whole_s plain_a/spec_kernel: 0.822 (range 0.781–0.906, n=4)
- decode_s_per_token plain_a/spec_kernel: 0.844 (range 0.802–0.933, n=4)
- decode_s plain_a/aspec_kernel: 1.140 (range 1.112–1.211, n=4)
- whole_s plain_a/aspec_kernel: 1.138 (range 1.112–1.209, n=4)
- decode_s_per_token plain_a/aspec_kernel: 1.023 (range 0.998–1.086, n=4)
- first divergence from plain_a (None = identical): {'plain_a': [None, None, None, None], 'spec': [None, None, None, None], 'aspec': [None, None, None, None], 'plain_b': [None, None, None, None], 'spec_kernel': [296, 296, 296, 296], 'aspec_kernel': [34, 34, 34, 34]}

## Verify cost: T-token forward over one-token forward

### compiled_kernel

- T2/T1: 1.517 (range 1.406–1.813, n=4)
- T3/T1: 1.447 (range 1.351–1.495, n=4)
- T5/T1: 1.519 (range 1.415–1.638, n=4)
- T9/T1: 1.788 (range 1.659–1.997, n=4)
- T13/T1: 1.647 (range 1.638–1.710, n=4)
- T17/T1: 3.239 (range 3.170–3.687, n=4)
- A/A first size, end/start: 0.986 (range 0.979–1.098, n=4)

### compiled

- T2/T1: 1.772 (range 1.468–1.841, n=4)
- T3/T1: 1.568 (range 1.472–1.672, n=4)
- T5/T1: 2.076 (range 1.681–2.208, n=4)
- T9/T1: 2.715 (range 2.492–2.826, n=4)
- T13/T1: 3.087 (range 2.732–3.222, n=4)
- T17/T1: 3.296 (range 3.108–3.495, n=4)
- A/A first size, end/start: 1.045 (range 0.874–1.122, n=4)

### eager

- T2/T1: 1.163 (range 1.034–1.249, n=4)
- T3/T1: 1.165 (range 1.064–1.439, n=4)
- T5/T1: 1.307 (range 1.082–1.566, n=4)
- T9/T1: 1.811 (range 1.564–1.989, n=4)
- T13/T1: 1.972 (range 1.657–2.350, n=4)
- T17/T1: 2.124 (range 1.774–2.235, n=4)
- A/A first size, end/start: 1.007 (range 0.803–1.074, n=4)

- rollback_compiled.json: pass
- rollback_compiled_kernel.json: pass
- rollback_eager.json: pass
- rollback_plant_skip_reset.json: fail
- rollback_plant_visible_sentinel.json: fail
- equiv_random.json: {'compiled': True, 'compiled_kernel': True}
- equiv_real.json: {'compiled': True, 'compiled_kernel': True}
