import sys

from train import main as train_main


if __name__ == "__main__":
    mode = "full"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in {"full", "fast"}:
            print(f"Unknown mode: {mode}. Use 'full' or 'fast'.", file=sys.stderr)
            sys.exit(2)
    train_main(mode)
