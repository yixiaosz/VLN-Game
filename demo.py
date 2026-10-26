#!/usr/bin/env python3

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PYTHON = Path.home() / "miniforge3/envs/vlngame/bin/python"
DEFAULT_GSA = Path.home() / "Grounded-Segment-Anything"
API_ENV = Path.home() / ".config/vlngame/api.env"

DEFAULT_MODELS = [
    {
        "name": "GPT-4o",
        "id": "openai/gpt-4o",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "supports_temperature": True,
        "options": {},
    },
    {
        "name": "GPT-4o mini",
        "id": "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "supports_temperature": True,
        "options": {},
    },
    {
        "name": "GPT-6 Luna",
        "id": "openai/gpt-6-luna",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "supports_temperature": False,
        "options": {"reasoning": {"effort": "none"}},
    },
    {
        "name": "DeepSeek V4.1 Flash",
        "id": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "supports_temperature": True,
        "options": {"thinking": {"type": "disabled"}},
    },
]

MODE_NAMES = {
    "objectnav": "ObjectNav mode (reference)",
    "clip": "Language mode: CLIP-based",
    "llm": "Language mode: VLM",
}
TOTALS = {"objectnav": 2000, "clip": 316, "llm": 316}
LOG_DIRS = {
    "objectnav": "objectnav-dino",
    "clip": "vlobjectnav-clip",
    "llm": "vlobjectnav-vlm_game",
}


def clear_screen():
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def ask_yes_no(prompt, default_yes=True):
    value = input(prompt).strip().lower()
    if not value:
        return default_yes
    return value in {"y", "yes"}


def load_models():
    raw = os.environ.get("VLN_MODELS")
    if not raw:
        return list(DEFAULT_MODELS)
    try:
        models = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Invalid VLN_MODELS JSON: {exc}")
        return None
    if not isinstance(models, list) or not models:
        print("VLN_MODELS must be a non-empty JSON list.")
        return None

    valid = []
    for index, model in enumerate(models, start=1):
        if not isinstance(model, dict):
            print(f"VLN_MODELS entry {index} must be an object.")
            return None
        for field in ("name", "id", "base_url", "key_env"):
            value = model.get(field)
            if not isinstance(value, str) or not value.strip():
                print(f"VLN_MODELS entry {index} is missing {field}.")
                return None
        options = model.get("options", {})
        if not isinstance(options, dict):
            print(f"VLN_MODELS entry {index} options must be an object.")
            return None
        supports_temperature = model.get("supports_temperature", True)
        if not isinstance(supports_temperature, bool):
            print(f"VLN_MODELS entry {index} supports_temperature must be true or false.")
            return None
        valid.append({
            "name": model["name"],
            "id": model["id"],
            "base_url": model["base_url"],
            "key_env": model["key_env"],
            "supports_temperature": supports_temperature,
            "options": options,
        })
    return valid


def choose_visualization():
    while True:
        clear_screen()
        print("Display mode")
        print()
        print("1) Headless (no window)")
        print("2) GUI (Open3D)")
        print("q) Quit")
        choice = input("\nSelect: ").strip().lower()
        if choice in {"1", "headless"}:
            return 0
        if choice in {"2", "gui"}:
            return 1
        if choice in {"q", "quit", "exit"}:
            return None
        print("Invalid choice.")
        input("Press Enter to continue...")


def choose_mode():
    while True:
        clear_screen()
        print("=" * 34)
        print("          VLN-Game demo")
        print("=" * 34)
        print("1) ObjectNav mode (reference)")
        print("2) Language mode: CLIP-based")
        print("3) Language mode: VLM")
        print("q) Quit")
        choice = input("\nSelect: ").strip().lower()
        if choice in {"1", "objectnav", "object"}:
            return "objectnav"
        if choice in {"2", "clip"}:
            return "clip"
        if choice in {"3", "llm", "vlm"}:
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


def choose_model(models):
    while True:
        clear_screen()
        print("Language / VLM model")
        print()
        for index, model in enumerate(models, start=1):
            print(f"{index}) {model['name']} ({model['id']})")
        print("b) Back")
        choice = input("\nSelect: ").strip().lower()
        if choice in {"b", "back"}:
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(models):
                return models[index - 1]
        print("Invalid choice.")
        input("Press Enter to continue...")


def show_summary(mode, count, model, visualize):
    clear_screen()
    print("Demo summary")
    print()
    print(f"Mode:     {MODE_NAMES[mode]}")
    if count == -1:
        print(f"Episodes: Full run ({TOTALS[mode]:,})")
    else:
        print(f"Episodes: {count}")
    if model is not None:
        print(f"Model:    {model['name']} ({model['id']})")
    print(f"Display:  {'GUI (Open3D)' if visualize else 'Headless (no window)'}")

    if count == -1:
        print()
        print("Full runs are long.")
        if mode == "llm":
            print("The VLM language run has API cost.")
        if not ask_yes_no("Continue with the full run? [y/N]: ", default_yes=False):
            return False

    print()
    return ask_yes_no("Launch? [Y/n]: ", default_yes=True)


def build_launch(mode, count, model, visualize):
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

    if visualize and not env.get("DISPLAY"):
        print("DISPLAY is not set.")
        print("Run the launcher from the desktop session with GUI access.")
        return None

    if mode == "llm":
        key_env = model["key_env"]
        api_key = env.get(key_env)
        if not api_key:
            print(f"{key_env} is not set.")
            print(f"Run: source {shlex.quote(str(API_ENV))}")
            return None
        env["VLN_API_KEY"] = api_key
        env["VLN_API_BASE"] = model["base_url"]
        env["VLN_MODEL"] = model["id"]
        env["VLN_MODEL_OPTIONS"] = json.dumps(model["options"])
        env["VLN_MODEL_SUPPORTS_TEMPERATURE"] = "1" if model["supports_temperature"] else "0"

    if mode == "objectnav":
        command = [str(python), "main_vis.py"]
    else:
        command = [str(python), "main_vln.py"]
        if mode == "clip":
            command.extend(["--vln_mode", "clip"])
    command.extend(["-v", str(visualize)])
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
    visualize = choose_visualization()
    if visualize is None:
        return 0

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
                models = load_models()
                if models is None:
                    input("Press Enter to return to the menu...")
                    continue
                model = choose_model(models)
                if model is None:
                    continue

            if not show_summary(mode, count, model, visualize):
                continue

            launch = build_launch(mode, count, model, visualize)
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
