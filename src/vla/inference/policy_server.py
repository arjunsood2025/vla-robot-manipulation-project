"""Async policy-inference server.

The GPU box runs the model; the robot machine (which may be a laptop tethered to
the arm) runs a thin client. They talk over a WebSocket: the client sends an
observation {images, state, instruction}, the server returns an action chunk plus
the server-side compute time. Decoupling inference from control this way means
(a) the robot loop never blocks on a slow GPU forward pass — it can request the
next chunk while executing the current one — and (b) the same trained model can
drive sim, a real arm, or the evaluation harness without change.

Latency is measured and returned on every request; end-to-end latency (capture ->
applied) is measured on the client. Reporting both is part of the deliverable.
"""
from __future__ import annotations

import argparse
import asyncio
import time

from vla.inference.serialization import pack, unpack


class PolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000):
        self.policy = policy
        self.host = host
        self.port = port

    async def _handle(self, websocket):
        async for message in websocket:
            request = unpack(message)
            t0 = time.perf_counter()
            chunk = self.policy.act(request)              # (H, action_dim)
            compute_ms = (time.perf_counter() - t0) * 1e3
            response = {
                "action_chunk": chunk,
                "server_compute_ms": compute_ms,
                "request_id": request.get("request_id", -1),
            }
            await websocket.send(pack(response))

    async def _serve(self):
        import websockets

        print(f"[policy_server] serving on ws://{self.host}:{self.port} "
              f"(action_dim={self.policy.action_dim})")
        async with websockets.serve(self._handle, self.host, self.port, max_size=None):
            await asyncio.Future()  # run forever

    def run(self):
        asyncio.run(self._serve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VLA policy inference server.")
    parser.add_argument("--policy-kind", required=True,
                        choices=["bc", "act", "smolvla", "pi0", "openvla"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .pt (bc) or checkpoint dir (lerobot/openvla).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from vla.inference.policy_wrapper import load_policy

    policy = load_policy(args.policy_kind, args.checkpoint, args.device)
    PolicyServer(policy, host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
