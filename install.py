"""
ComfyUI-Manager install hook for ai-toolkit-inference.

This script is automatically executed by ComfyUI-Manager when installing or
updating this custom node pack. It:
1. Installs Python dependencies from requirements-inference.txt
2. Clones (or updates) ostris/ai-toolkit into vendor/ai-toolkit

After installation, extended models (FLUX.2, Chroma, HiDream, OmniGen2, LTX-2,
Wan 2.2) will work without manually setting AI_TOOLKIT_PATH.
"""

import os
import subprocess
import sys

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIR, "requirements-inference.txt")
VENDOR_DIR = os.path.join(SCRIPT_DIR, "vendor")
AI_TOOLKIT_DIR = os.path.join(VENDOR_DIR, "ai-toolkit")
AI_TOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"


def run_command(cmd: list, cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print output."""
    print(f"[ai-toolkit-inference] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def install_requirements():
    """Install Python dependencies from requirements-inference.txt."""
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"[ai-toolkit-inference] Warning: {REQUIREMENTS_FILE} not found, skipping pip install")
        return

    print("[ai-toolkit-inference] Installing Python dependencies...")
    try:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                REQUIREMENTS_FILE,
                "--quiet",
            ]
        )
        print("[ai-toolkit-inference] Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"[ai-toolkit-inference] Warning: pip install failed (exit code {e.returncode})")
        print("[ai-toolkit-inference] You may need to manually run:")
        print(f"    pip install -r {REQUIREMENTS_FILE}")


def clone_or_update_ai_toolkit():
    """Clone or update ostris/ai-toolkit into vendor/ai-toolkit."""
    # Check if git is available
    try:
        run_command(["git", "--version"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ai-toolkit-inference] Warning: git not found")
        print("[ai-toolkit-inference] Extended models (FLUX.2, Chroma, HiDream, OmniGen2, LTX-2, Wan 2.2)")
        print("                       require ai-toolkit. Please install git and re-run, or manually clone:")
        print(f"    git clone {AI_TOOLKIT_REPO} {AI_TOOLKIT_DIR}")
        return

    # Create vendor directory if needed
    os.makedirs(VENDOR_DIR, exist_ok=True)

    if os.path.isdir(os.path.join(AI_TOOLKIT_DIR, ".git")):
        # Already cloned - pull latest
        print("[ai-toolkit-inference] Updating ai-toolkit...")
        try:
            run_command(["git", "pull", "--ff-only"], cwd=AI_TOOLKIT_DIR)
            print("[ai-toolkit-inference] ai-toolkit updated successfully")
        except subprocess.CalledProcessError:
            print("[ai-toolkit-inference] Warning: git pull failed, using existing version")
    else:
        # Fresh clone
        print("[ai-toolkit-inference] Cloning ai-toolkit (this may take a moment)...")
        # Remove any partial/non-git directory
        if os.path.exists(AI_TOOLKIT_DIR):
            import shutil

            shutil.rmtree(AI_TOOLKIT_DIR)
        try:
            run_command(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",  # Shallow clone for faster download
                    AI_TOOLKIT_REPO,
                    AI_TOOLKIT_DIR,
                ]
            )
            print("[ai-toolkit-inference] ai-toolkit cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ai-toolkit-inference] Error: git clone failed (exit code {e.returncode})")
            print("[ai-toolkit-inference] Extended models will not work until ai-toolkit is available.")
            print("[ai-toolkit-inference] You can manually clone with:")
            print(f"    git clone {AI_TOOLKIT_REPO} {AI_TOOLKIT_DIR}")


def main():
    print("=" * 60)
    print("[ai-toolkit-inference] Starting installation...")
    print("=" * 60)

    install_requirements()
    clone_or_update_ai_toolkit()

    print("=" * 60)
    print("[ai-toolkit-inference] Installation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
