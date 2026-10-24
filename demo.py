#!/usr/bin/env python3

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PYTHON = Path.home() / "miniforge3/envs/vlngame/bin/python"
DEFAULT_GSA = Path.home() / "Grounded-Segment-Anything"
DEEPSEEK_ENV = Path.home() / ".config/vlngame/deepseek.env"
MODELS = [("DeepSeek", "deepseek-flash")]

MODE_NAMES = {"objectnav": "ObjectNav", "llm": "Language / VLM"}
TOTALS = {"objectnav": 2000, "llm": 316}
LOG_DIRS = {"objectnav": "objectnav-dino", "llm": "vlobjectnav-vlm_game"}


def clear_screen():
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def ask_yes_no(prompt, default_yes=True):
    value = input(prompt).strip().lower()
    if not value:
        return default_yes
    return value in {"y", "yes"}


def choose_mode():
    while True:
        clear_screen()
        print("=" * 34)
        print("          VLN-Game demo")
        print("=" * 34)
        print("1) ObjectNav")
        print("2) Language / VLM")
        print("q) Quit")
        choice = input("\nSelect: ").strip().lower()
        if choice in {"1", "objectnav", "object"}:
            return "objectnav"
        if choice in {"2", "llm", "language", "vlm"}:
            return "llm"
        if choice in {"q", "quit", "exit"}:
            return None
        print("Invalid choice.")
        input("Press Enter to continue...")


def choose_count(mode):
    total = TOTALS[mode]
    while True:
        clear_screen()
        print(f"{MODE_NAMES[mode]} episodes")
        print()
        print("1) 5 episodes")
        print(f"2) Full run ({total:,} episodes)")
        print("3) Custom count")
        print("b) Back")
        choice = input("\nSelect: ").strip().lower()
        if choice == "1":
            return 5
        if choice == "2":
            return -1
        if choice == "3":
            while True:
                value = input(f"Episode count (1-{total}): ").strip()
                if value.lower() in {"", "b", "back"}:
                    break
                try:
                    count = int(value)
                except ValueError:
                    print(f"Enter a number from 1 to {total}.")
                    continue
                if 1 <= count <= total:
                    return count
                print(f"Enter a number from 1 to {total}.")
        elif choice in {"b", "back"}:
            return None
        else:
            print("Invalid choice.")
            input("Press Enter to continue...")


def choose_model():
    while True:
        clear_screen()
        print("Language / VLM model")
        print()
        for index, (name, model_id) in enumerate(MODELS, start=1):
            print(f"{index}) {name} ({model_id})")
        print("b) Back")
        choice = input("\nSelect: ").strip().lower()
        if choice in {"b", "back"}:
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(MODELS):
                return MODELS[index - 1]
        print("Invalid choice.")
        input("Press Enter to continue...")


def show_summary(mode, count, model):
    clear_screen()
    print("Demo summary")
    print()
    print(f"Mode:     {MODE_NAMES[mode]}")
    if count == -1:
        print(f"Episodes: Full run ({TOTALS[mode]:,})")
    else:
        print(f"Episodes: {count}")
    if model is not None:
        print(f"Model:    {model[0]} ({model[1]})")
    print("GUI:      Open3D")

    if count == -1:
        print()
        print("Full runs are long and the language run has API cost.")
        if not ask_yes_no("Continue with the full run? [y/N]: ", default_yes=False):
            return False

    print()
    return ask_yes_no("Launch? [Y/n]: ", default_yes=True)


def build_launch(mode, count, model):
    env = os.environ.copy()
    gsa = Path(env.get("GSA_PATH", DEFAULT_GSA)).expanduser().resolve()
    if not gsa.is_dir():
        print(f"GSA_PATH does not exist: {gsa}")
        print("Set GSA_PATH to the Grounded-Segment-Anything checkout.")
        return None
    env["GSA_PATH"] = str(gsa)

    if ENV_PYTHON.is_file():
        python = ENV_PYTHON
        env.setdefault("CUDA_HOME", str(ENV_PYTHON.resolve().parent.parent))
    else:
        python = Path(sys.executable)

    if not env.get("DISPLAY"):
        print("DISPLAY is not set.")
        print("Run the launcher from the desktop session with GUI access.")
        return None

    if mode == "llm":
        if not env.get("DEEPSEEK_API_KEY"):
            print("DEEPSEEK_API_KEY is not set.")
            print(f"Run: source {shlex.quote(str(DEEPSEEK_ENV))}")
            return None
        env.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        env["VLN_MODEL"] = model[1]

    script = "main_vis.py" if mode == "objectnav" else "main_vln.py"
    command = [str(python), script, "-v", "1"]
    if count != -1:
        command.extend(["--episode_count", str(count)])
    return command, env


def run_demo(mode, command, env):
    log_path = ROOT / "dump" / "logs" / LOG_DIRS[mode] / "eval.log"
    print()
    print("Launching:")
    print(" ".join(shlex.quote(part) for part in command))
    print()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return
    except OSError as exc:
        print(f"Failed to launch: {exc}")
        return

    print()
    print(f"Demo exited with status {result.returncode}.")
    print(f"Logs: {log_path}")


def main():
    while True:
        mode = choose_mode()
        if mode is None:
            return 0

        while True:
            count = choose_count(mode)
            if count is None:
                break

            model = None
            if mode == "llm":
                model = choose_model()
                if model is None:
                    continue

            if not show_summary(mode, count, model):
                continue

            launch = build_launch(mode, count, model)
            if launch is None:
                input("Press Enter to return to the menu...")
                continue

            command, env = launch
            run_demo(mode, command, env)
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        raise SystemExit(1)
