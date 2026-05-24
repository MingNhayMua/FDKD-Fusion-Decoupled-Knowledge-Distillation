"""
FDKD Demo — Google Colab Launcher
"""
import os
import sys
import argparse

# Add project root to path so all modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def start_server(
    checkpoint_dir: str = "/content/drive/MyDrive/checkpoints",
    ngrok_token: str = "",
    port: int = 8000,
    labels_path: str = None,
):
    """Start the FDKD demo server on Colab with ngrok tunnel."""
    import nest_asyncio
    import asyncio
    from pyngrok import ngrok

    import utils.config as cfg
    cfg.CHECKPOINT_DIR = checkpoint_dir

    from utils.labels import load_labels
    lp = labels_path or os.path.join(checkpoint_dir, "tiny_imagenet_labels.json")
    load_labels(lp)

    from backend.models.loader import load_all_models
    load_all_models()

    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)

    public_url = ngrok.connect(port)
    print("\n" + "=" * 60)
    print(f"🚀 FDKD Demo Server is running!")
    print(f"📡 Public URL: {public_url}")
    print(f"🔗 Paste this URL in the frontend connection input")
    print("=" * 60 + "\n")

    nest_asyncio.apply()
    import uvicorn
    config = uvicorn.Config(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    asyncio.get_event_loop().run_until_complete(server.serve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDKD Demo Server")
    parser.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/checkpoints")
    parser.add_argument("--token", default="", help="ngrok auth token")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--labels", default=None, help="Path to labels JSON")
    args = parser.parse_args()

    start_server(
        checkpoint_dir=args.checkpoint_dir,
        ngrok_token=args.token,
        port=args.port,
        labels_path=args.labels,
    )
