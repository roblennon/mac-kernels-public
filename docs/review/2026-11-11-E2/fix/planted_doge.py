"""Planted fault: the T = 1 clause unconditional (the reviewed bug). The two new tests must fail."""
import sys; sys.path.insert(0, "<repo>")
from bench import smallq_hf
import tests.test_smallq_hf as t
orig = smallq_hf.enable
def enable(model, threshold=4, decode_offset=False):
    r = orig(model, threshold); smallq_hf.SETTINGS["decode_offset"] = True; return r
smallq_hf.enable = enable
class Forced(dict):
    def __getitem__(self, k): return True if k == "decode_offset" else dict.__getitem__(self, k)
for name in ("test_model_that_indexes_its_mask_is_stock_at_threshold_one", "test_a_requested_materialized_mask_is_kept_unless_opted_in"):
    smallq_hf.SETTINGS = Forced(threshold=4, decode_offset=False)
    try:
        getattr(t, name)(); print(name, "PASSED (fault not caught)")
    except Exception as e:
        print(name, "FAILED as required:", type(e).__name__, str(e)[:120])
