"""PreToolUse hook: deny Bash/PowerShell commands that call a bare python/py/pip.

Bare names resolve to the system install or the Microsoft Store stub on this machine.
Full-path calls to the miniforge `behavython` env (or any other full path) are allowed.
"""

import json
import re
import sys

# The hook runs under the behavython env, so its own interpreter is the one to recommend.
ENV_PYTHON = sys.executable
ENV_PYTHON_BASH = "/" + ENV_PYTHON[0].lower() + ENV_PYTHON[2:].replace("\\", "/")

# Start of a command: line start, a shell separator, a subshell, or a command-prefixing word.
COMMAND_START = r"(?:^|[;&|\n(`{]|\$\(|\b(?:then|do|else|xargs|env|nohup|exec|time|start)\s)"
ENV_ASSIGNMENTS = r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
BARE_INTERPRETER = r"(python(?:\d+(?:\.\d+)?)?|py|pip\d*(?:\.\d+)?)(?:\.exe)?"
PATTERN = re.compile(
    COMMAND_START + r"\s*" + ENV_ASSIGNMENTS + BARE_INTERPRETER + r"(?=\s|$|[;&|)`])",
    re.IGNORECASE | re.MULTILINE,
)


def main() -> int:
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8") or "{}")
    command = payload.get("tool_input", {}).get("command", "")
    match = PATTERN.search(command)
    if not match:
        return 0

    reason = (
        f"Blocked bare `{match.group(1)}`: system-wide Python is not allowed in this repo (see AGENTS.md section 3). "
        f"Use the full path `{ENV_PYTHON}` (Git Bash: {ENV_PYTHON_BASH}), "
        "run pip as `<env python> -m pip`, and set PYTHONPATH=src for project code."
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
