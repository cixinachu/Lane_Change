"""
Quick inspection/visualization tool for expert_data.pkl.

Usage:
    python inspect_expert.py --path runs/log/reward_recovery_new/main2_SocialAIRL_20251211-100531/expert_data.pkl
    python inspect_expert.py --path ... --plot
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path, help="Path to expert_data.pkl")
    parser.add_argument("--plot", action="store_true", help="Plot first trajectory (requires matplotlib)")
    parser.add_argument("--reconstruct_eps", action="store_true", help="Reconstruct episodes using dones and plot first one")
    args = parser.parse_args()

    with args.path.open("rb") as f:
        data = pickle.load(f)

    print(f"Loaded: {args.path}")
    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        for k, v in data.items():
            arr = np.array(v)
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")
    else:
        arr = np.array(data)
        print(f"  data: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"matplotlib not available: {e}")
            return

        # Try to find a reasonable trajectory array
        traj = None
        if isinstance(data, dict):
            for key in ("trajectories", "traj", "obs", "observations"):
                if key in data:
                    arr = np.array(data[key])
                    if arr.ndim >= 3:
                        traj = arr[0]
                        break
        else:
            arr = np.array(data)
            if arr.ndim >= 3:
                traj = arr[0]

        if traj is not None:
            # Try a few projections to avoid overlap (assume state layout: [*feat, y, x, vx, vy] or similar)
            proj_candidates = [
                ("last2", traj[..., -2], traj[..., -1]),          # default
                ("y_x", traj[..., -4], traj[..., -3]),            # y, x if last 4 are [y,x,vx,vy]
                ("feat01", traj[..., 0], traj[..., 1]),           # first two features
            ]
            plt.figure(figsize=(12, 4))
            for idx, (name, x, y) in enumerate(proj_candidates, 1):
                plt.subplot(1, 3, idx)
                plt.plot(x, y, "-o", markersize=2)
                plt.title(f"Traj0 proj {name}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No trajectory-like array found to plot (trajectories/obs not found).")

    # Reconstruct episodes using dones
    if args.reconstruct_eps and isinstance(data, dict) and "dones" in data and "state" in data:
        dones = np.array(data["dones"]).squeeze()
        states = np.array(data["state"])
        actions = np.array(data.get("action", []))
        episodes = []
        start = 0
        for i, done in enumerate(dones):
            if done:
                episodes.append((states[start:i + 1], actions[start:i + 1] if actions.size else None))
                start = i + 1
        print(f"Reconstructed {len(episodes)} episodes.")
        if episodes and args.plot:
            s0, a0 = episodes[0]
            proj_candidates = [
                ("last2", s0[..., -2], s0[..., -1]),
                ("y_x", s0[..., -4], s0[..., -3]),
                ("feat01", s0[..., 0], s0[..., 1]),
            ]
            plt.figure(figsize=(12, 4))
            for idx, (name, x, y) in enumerate(proj_candidates, 1):
                plt.subplot(1, 3, idx)
                plt.plot(x, y, "-o", markersize=2)
                plt.title(f"Episode0 proj {name}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
