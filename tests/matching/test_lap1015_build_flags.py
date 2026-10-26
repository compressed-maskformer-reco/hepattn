"""The default lap1015 build must run on any CPU of its architecture, not only on the build host's.

CI restores the compiled extension from a pixi-environment cache that every GitHub runner shares.
With `-march=native` in the default flags, a cache saved on an AVX-512 runner crashed the unit job
with "Fatal Python error: Illegal instruction" on `import lap1015` whenever a runner without
AVX-512 restored it. A crash cannot be reproduced on demand (it depends on which CPUs the jobs
get), so this checks the cause in `CMakeLists.txt` instead.
"""

import re
from pathlib import Path

CMAKELISTS = Path(__file__).resolve().parents[2] / "CMakeLists.txt"

# Flags that raise the instruction set above the architecture baseline. -mtune only changes
# scheduling and is safe; -m64 and -mfpmath=sse are part of the x86-64 baseline.
ISA_FLAG = re.compile(r"-m(?:arch|cpu)=|-mavx|-mfma|-mbmi|-msse4|-maes|-mpclmul|-mpopcnt")
OPT_IN = "LAP1015_NATIVE"


def _code_lines(text: str) -> list[str]:
    # CMake comments start with '#'; this file has no '#' inside strings.
    return [line.split("#", 1)[0].strip() for line in text.splitlines()]


def _isa_flags_by_branch(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return (line, enclosing branches) for every line that sets an ISA flag.

    A branch is the condition of the enclosing if(), or "else:<condition>" for the else() and
    elseif() parts, so a flag in the else() of if(LAP1015_NATIVE) does not count as opted in.
    option() and message() lines are skipped: their text may name a flag without setting it.
    """
    stack: list[str] = []
    found = []
    for line in _code_lines(text):
        command = line.split("(", 1)[0].strip().lower()
        argument = line.split("(", 1)[1].rsplit(")", 1)[0].strip() if "(" in line else ""
        if command == "if":
            stack.append(argument)
        elif command in {"elseif", "else"}:
            stack[-1] = "else:" + stack[-1].removeprefix("else:")
        elif command == "endif":
            stack.pop()
        elif command not in {"option", "message"} and ISA_FLAG.search(line):
            found.append((line, tuple(stack)))
    assert not stack, f"unbalanced if()/endif() in {CMAKELISTS}"
    return found


def test_default_flags_are_portable():
    offending = [line for line, branches in _isa_flags_by_branch(CMAKELISTS.read_text()) if OPT_IN not in branches]
    assert not offending, (
        f"{CMAKELISTS.name} sets CPU-specific flags outside if({OPT_IN}): {offending}. Such a build "
        "crashes with 'Illegal instruction' on a CPU without the build host's instruction set, which "
        f"is how cached CI environments broke. Put the flag inside if({OPT_IN})."
    )


def test_native_build_is_opt_in_and_off_by_default():
    text = "\n".join(_code_lines(CMAKELISTS.read_text()))
    assert re.search(rf'option\(\s*{OPT_IN}\s+"[^"]*"\s+OFF\s*\)', text), f"option({OPT_IN} ... OFF) is missing"
    opted_in = [line for line, branches in _isa_flags_by_branch(text) if OPT_IN in branches]
    assert any("-march=native" in line for line in opted_in), f"-DLAP1015_NATIVE=ON no longer adds -march=native: {opted_in}"
