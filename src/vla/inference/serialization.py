"""msgpack (de)serialization for observation/action messages on the wire.

We use msgpack + msgpack-numpy rather than JSON because observations carry raw
camera frames — encoding a 640x480x3 array as JSON numbers is ~10x larger and
much slower than msgpack's binary packing, and image latency is on the critical
path of the 30 Hz control loop. Keeping the codec in one module means the server
and client can never drift out of sync on the format.
"""
from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy as m

m.patch()  # teach msgpack to (un)pack numpy arrays


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def unpack(blob: bytes) -> Any:
    return msgpack.unpackb(blob, raw=False)
