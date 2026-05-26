"""
FDKD Demo — Local Server Launcher

Run the backend locally on Mac/Linux without Colab or ngrok.
The frontend at localhost:3000 connects directly to localhost:8000.

Usage:
    python run_local.py
    python run_local.py --checkpoint-dir /path/to/checkpoints --port 8000
"""
import os
import sys
import argparse

# Add project root to path so all modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def start_server(
    checkpoint_dir: str = None,
    port: int = 8000,
    labels_path: str = None,
):
    """Start the FDKD demo server locally (no ngrok needed)."""
    import utils.config as cfg

    if checkpoint_dir:
        cfg.CHECKPOINT_DIR = checkpoint_dir

    print(f"\n{'='*60}")
    print(f"🖥  FDKD Demo — Local Mode")
    print(f"{'='*60}")
    print(f"  Device:      {cfg.DEVICE}")
    print(f"  Checkpoints: {cfg.CHECKPOINT_DIR}")
    print(f"  Port:        {port}")

    if not os.path.isdir(cfg.CHECKPOINT_DIR):
        print(f"\n  ⚠️  Checkpoint directory not found: {cfg.CHECKPOINT_DIR}")
        print(f"  Create it and copy .pth files there, or set --checkpoint-dir")
        print(f"  Models will load without weights (random predictions)\n")

    # Load labels
    from utils.labels import load_labels
    lp = labels_path or os.path.join(cfg.CHECKPOINT_DIR, "tiny_imagenet_labels.json")
    if not os.path.exists(lp):
        # Try project-local copy
        lp_local = os.path.join(PROJECT_ROOT, "utils", "tiny_imagenet_labels.json")
        if os.path.exists(lp_local):
            lp = lp_local
    load_labels(lp)

    # Load models
    from backend.models.loader import load_all_models
    load_all_models()

    print(f"\n{'='*60}")
    print(f"🚀 Server running at: http://localhost:{port}")
    print(f"🔗 Frontend at:       http://localhost:3000")
    print(f"📡 API docs at:       http://localhost:{port}/docs")
    print(f"{'='*60}\n")

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDKD Demo — Local Server")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to checkpoint directory (auto-detected if not set)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--labels", default=None, help="Path to labels JSON")
    args = parser.parse_args()

    start_server(
        checkpoint_dir=args.checkpoint_dir,
        port=args.port,
        labels_path=args.labels,
    )
