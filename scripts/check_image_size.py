#!/usr/bin/env python3
"""
Check Docker image size against the project budget.

Budget (from cicd-and-deployment.md):
  Preferred:  200–350 MB
  Maximum:    500 MB
  Hard fail:  > 500 MB

Usage:
    python scripts/check_image_size.py <image_name_or_tag>
    python scripts/check_image_size.py llm-eval-cicd:latest
"""
import subprocess
import sys


PREFERRED_MAX_MB = 350
HARD_MAX_MB = 500


def get_image_size_mb(image_tag: str) -> float:
    """Return Docker image size in megabytes."""
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag, "--format", "{{.Size}}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"ERROR: Could not inspect image '{image_tag}'")
        print(f"  {result.stderr.strip()}")
        sys.exit(2)
    size_bytes = int(result.stdout.strip())
    return size_bytes / (1024 * 1024)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_image_size.py <image_tag>")
        sys.exit(2)

    image_tag = sys.argv[1]
    size_mb = get_image_size_mb(image_tag)

    print(f"\n{'='*50}")
    print(f"Docker Image Size Check")
    print(f"Image:    {image_tag}")
    print(f"Size:     {size_mb:.1f} MB")
    print(f"Budget:   {PREFERRED_MAX_MB}–{HARD_MAX_MB} MB")
    print(f"{'='*50}")

    if size_mb <= PREFERRED_MAX_MB:
        print(f"✓ PASS — {size_mb:.1f} MB is within preferred range (≤{PREFERRED_MAX_MB} MB)")
        sys.exit(0)
    elif size_mb <= HARD_MAX_MB:
        print(f"⚠ WARN — {size_mb:.1f} MB exceeds preferred ({PREFERRED_MAX_MB} MB) but within maximum ({HARD_MAX_MB} MB)")
        print("  Tip: Remove unused dependencies to reduce image size.")
        sys.exit(0)
    else:
        print(f"✗ FAIL — {size_mb:.1f} MB exceeds hard maximum of {HARD_MAX_MB} MB")
        print("  The production environment has upload bandwidth constraints.")
        print("  Remove PyTorch, TensorFlow, CUDA, model weights, or unused large packages.")
        sys.exit(1)


if __name__ == "__main__":
    main()
