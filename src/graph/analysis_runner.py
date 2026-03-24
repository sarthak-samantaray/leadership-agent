"""LLM-generated pandas/matplotlib charts in a temp dir; returns PNG paths for HTML embedding."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.logger import get_logger
from src.utils.message_content import message_content_to_str
from src.utils.prompts import PYTHON_VIZ_SYSTEM

log = get_logger(__name__)

_FORBIDDEN = re.compile(
    r"\b(subprocess|os\.system|eval\s*\(|exec\s*\(|__import__|socket|requests\.|urllib|httpx)\b",
    re.IGNORECASE,
)
_SEABORN_PALETTE_FW = re.compile(
    r"[^\n]*_viz_run\.py:\d+:\s*FutureWarning:\s*\n\n"
    r"Passing `palette` without assigning `hue` is deprecated[\s\S]*?\n\n"
    r"\s*sns\.(?:barplot|countplot)\([^\n]*\)\s*\n?",
)


def _strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    blocks = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE
    )
    if blocks:
        return max(blocks, key=len).strip()
    text = re.sub(r"^```\s*python\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _fix_seaborn_barplot_palette_hue(code: str) -> str:
    out_lines: list[str] = []
    for line in code.splitlines():
        if not re.search(r"sns\.(barplot|countplot)\s*\(", line):
            out_lines.append(line)
            continue
        if "palette=" not in line or "hue=" in line:
            out_lines.append(line)
            continue
        m = re.search(r"\bx\s*=\s*([^,)\n]+)", line)
        if not m:
            out_lines.append(line)
            continue
        xval = m.group(1).strip()
        line = line.replace("palette=", f"hue={xval}, palette=", 1)
        if "legend=False" not in line:
            ridx = line.rfind(")")
            if ridx != -1:
                line = line[:ridx] + ", legend=False" + line[ridx:]
        out_lines.append(line)
    return "\n".join(out_lines)


def _clean_stderr(stderr: str) -> str:
    if not stderr:
        return ""
    cleaned = stderr
    while True:
        nxt = _SEABORN_PALETTE_FW.sub("", cleaned, count=1)
        if nxt == cleaned:
            break
        cleaned = nxt
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


_VIZ_COMPILE_RETRIES = 3


def run_python_viz_analysis(
    llm: BaseChatModel,
    user_query: str,
    evidence: str,
    reports_dir: Path,
    timeout_seconds: int = 90,
) -> tuple[str, str, list[str]]:
    """Returns (stdout, stderr, png_paths absolute)."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    evidence = (evidence or "")[:14000]
    base_human = (
        f"User request:\n{user_query}\n\n"
        f"Context / numbers / excerpts to use:\n{evidence or '(none)'}\n\n"
        "Output the full Python script only."
    )
    last_syntax_err = ""
    code = ""
    for attempt in range(_VIZ_COMPILE_RETRIES):
        if attempt == 0:
            human = base_human
        else:
            prev = code[:12000] if code.strip() else "(empty)"
            human = (
                f"{base_human}\n\n---\n"
                f"Your previous output was not valid Python (compile error): {last_syntax_err}\n"
                "Return a COMPLETE script only: close every [, (, {{, and string quote.\n\n"
                f"Broken script to fix or replace:\n```python\n{prev}\n```"
            )
        resp = llm.invoke(
            [SystemMessage(content=PYTHON_VIZ_SYSTEM), HumanMessage(content=human)]
        )
        raw = message_content_to_str(
            resp.content if hasattr(resp, "content") else resp
        )
        code = _strip_code_fences(raw)
        code = _fix_seaborn_barplot_palette_hue(code)
        if not code.strip():
            last_syntax_err = "(empty code from model)"
            log.warning("viz codegen empty attempt %s", attempt + 1)
            continue
        if _FORBIDDEN.search(code):
            return "", "Blocked: disallowed pattern in script.", []
        preamble = (
            "import os\n"
            "import matplotlib\n"
            'matplotlib.use("Agg")\n'
        )
        if "matplotlib.use" not in code:
            code = preamble + code
        try:
            compile(code, "<viz_script>", "exec")
        except SyntaxError as e:
            last_syntax_err = str(e)
            log.warning("viz SyntaxError attempt %s: %s", attempt + 1, e)
            continue
        break
    else:
        return (
            "",
            f"Syntax error after {_VIZ_COMPILE_RETRIES} attempts: {last_syntax_err}",
            [],
        )

    with tempfile.TemporaryDirectory(prefix="viz_", dir=str(reports_dir)) as tmp:
        tmp_path = Path(tmp)
        script_path = tmp_path / "_viz_run.py"
        script_path.write_text(code, encoding="utf-8")
        env = os.environ.copy()
        env["VIZ_OUT_DIR"] = str(tmp_path)
        env["MPLBACKEND"] = "Agg"
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                cwd=str(tmp_path),
            )
            out, err = proc.stdout or "", _clean_stderr(proc.stderr or "")
            if proc.returncode != 0:
                err = f"exit {proc.returncode}\n{err}".strip()
            pngs = sorted(str(p) for p in tmp_path.glob("*.png"))
            persist = reports_dir / f"viz_assets_{tmp_path.name}"
            persist.mkdir(parents=True, exist_ok=True)
            saved: list[str] = []
            for p in pngs:
                dest = persist / Path(p).name
                dest.write_bytes(Path(p).read_bytes())
                saved.append(str(dest.resolve()))
            return out, err, saved
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {timeout_seconds}s", []
        except Exception as e:
            log.exception("viz run failed: %s", e)
            return "", str(e), []
