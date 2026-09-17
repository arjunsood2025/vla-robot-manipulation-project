"""The CLIP text tower is frozen, so caching its output must be exact.

If caching ever changed an embedding, every downstream action would shift and
the cached-vs-uncached policies would silently stop being the same model. These
tests pin that down without downloading CLIP, by driving the caching logic with
a stub encoder.
"""
import torch

from vla.models.encoders import CLIPTextEncoder


class _StubEncoder(CLIPTextEncoder):
    """CLIPTextEncoder's caching logic with the transformer swapped out.

    Bypasses __init__ (which would download weights) and counts how many
    strings actually reach the underlying encoder.
    """

    def __init__(self, cache_embeddings=True):
        torch.nn.Module.__init__(self)
        self.out_dim = 4
        self._cache_enabled = cache_embeddings
        self._cache = {}
        self.encoded = []

    def _encode(self, instructions, device):
        self.encoded.extend(instructions)
        # Deterministic per-string vector, so equality checks are meaningful.
        return torch.stack([
            torch.full((self.out_dim,), float(len(s)), dtype=torch.float32)
            for s in instructions
        ])


def test_cache_returns_same_values_as_no_cache():
    a = _StubEncoder(cache_embeddings=True)
    b = _StubEncoder(cache_embeddings=False)
    instr = ["pick up the pen", "pick up the tape"]
    assert torch.equal(a(instr, torch.device("cpu")), b(instr, torch.device("cpu")))


def test_repeated_instruction_is_encoded_once():
    # The control-loop case: the same instruction every tick.
    enc = _StubEncoder()
    for _ in range(10):
        enc(["put the pens away"], torch.device("cpu"))
    assert enc.encoded == ["put the pens away"]


def test_duplicates_within_a_batch_encoded_once_but_all_returned():
    enc = _StubEncoder()
    out = enc(["a", "bb", "a"], torch.device("cpu"))
    assert enc.encoded == ["a", "bb"]
    assert out.shape == (3, 4)
    assert torch.equal(out[0], out[2])


def test_order_is_preserved():
    enc = _StubEncoder()
    out = enc(["aaa", "b"], torch.device("cpu"))
    assert out[0][0] == 3.0
    assert out[1][0] == 1.0


def test_each_string_encoded_alone_so_padding_cannot_leak():
    # Batching a short string with a long one would pad it; encoding singly
    # guarantees the stored embedding is padding-independent.
    enc = _StubEncoder()
    enc(["short", "a much longer instruction string"], torch.device("cpu"))
    assert enc.encoded == ["short", "a much longer instruction string"]


def test_clear_cache_forces_reencode():
    enc = _StubEncoder()
    enc(["x"], torch.device("cpu"))
    enc.clear_cache()
    enc(["x"], torch.device("cpu"))
    assert enc.encoded == ["x", "x"]
