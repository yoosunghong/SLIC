"""Run the LLM micro-experiment via the Gemini REST API.

Usage:
    python -m llm_micro.runner --samples 15 --model gemini-2.5-flash
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from llm_micro.scenarios import (
    EXPLICIT_LABELS,
    IMPLICIT_LABELS,
    Scenario,
    build_prompt,
)

DISPATCH_RE = re.compile(r"DISPATCH:\s*([A-Za-z_]+)", re.IGNORECASE)
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
INVALID_KEY_MARKERS = ("API_KEY_INVALID", "API key expired", "Please renew the API key")


def parse_dispatch(text: str, condition: str) -> str | None:
    valid = EXPLICIT_LABELS if condition == "explicit" else IMPLICIT_LABELS
    m = DISPATCH_RE.search(text or "")
    if m:
        name = m.group(1).strip()
        for v in valid:
            if name.lower() == v.lower():
                return v
    lowered = (text or "").lower()
    for v in valid:
        if v.lower() in lowered:
            return v
    return None


def extract_text_from_response(data: dict) -> str:
    texts = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def parse_json_dispatch(text: str, condition: str) -> str | None:
    valid = EXPLICIT_LABELS if condition == "explicit" else IMPLICIT_LABELS
    if not text:
        return None
    candidates = [text]
    m = JSON_BLOCK_RE.search(text)
    if m:
        candidates.insert(0, m.group(0))
    for chunk in candidates:
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        dispatch = payload.get("dispatch")
        if not isinstance(dispatch, str):
            continue
        for name in valid:
            if dispatch.lower() == name.lower():
                return name
    return None


def mask_key(value: str) -> str:
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def build_request_prompt(prompt: str, condition: str, last_raw: str = "") -> str:
    valid = EXPLICIT_LABELS if condition == "explicit" else IMPLICIT_LABELS
    valid_names = ", ".join(valid)
    retry_suffix = ""
    if last_raw:
        retry_suffix = (
            "\nThe previous answer was invalid or incomplete.\n"
            f"Previous answer: {last_raw[:120]}\n"
        )
    return (
        f"{prompt}\n"
        f"{retry_suffix}"
        "Return JSON only with this exact schema:\n"
        '{"dispatch":"<name>","reason":"<short sentence>"}\n'
        f'The "dispatch" value must be exactly one of: {valid_names}.\n'
        "Do not use markdown. Do not omit the dispatch value.\n"
    )


def generate_content_rest(api_key: str, model_name: str, prompt: str) -> dict:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 128,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def run(samples_per_cell: int, model_name: str, out_path: Path, max_retries: int = 3):
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    print(f"Using GEMINI_API_KEY={mask_key(api_key)} from .env (override=True)")

    cells = []
    for context_role in ["Strike", "Vanguard", "Support"]:
        for formation in ["Clustered", "Spread"]:
            for condition in ["explicit", "implicit"]:
                cells.append((context_role, formation, condition))

    results = []
    total = len(cells) * samples_per_cell
    pbar = tqdm(total=total, desc="LLM calls")

    for context_role, formation, condition in cells:
        for seed in range(samples_per_cell):
            scenario = Scenario(context_role, formation, condition, seed)
            prompt = build_prompt(scenario)
            surface_to_role = scenario.surface_to_role()
            expected_dispatch = scenario.expected_dispatch()

            dispatched = None
            raw = ""
            for attempt in range(max_retries):
                try:
                    request_prompt = build_request_prompt(
                        prompt,
                        condition,
                        last_raw=raw if attempt > 0 else "",
                    )
                    payload = generate_content_rest(api_key, model_name, request_prompt)
                    raw = extract_text_from_response(payload)
                    dispatched = parse_json_dispatch(raw, condition) or parse_dispatch(raw, condition)
                    if dispatched is not None:
                        break
                except urllib.error.HTTPError as e:
                    error_body = e.read().decode("utf-8", errors="replace")
                    raw = f"ERROR: {error_body}"
                    if any(marker in raw for marker in INVALID_KEY_MARKERS):
                        raise RuntimeError(
                            "Gemini rejected the API key before sampling started. "
                            "Check whether your shell has a stale GEMINI_API_KEY and "
                            "confirm the intended key is loaded from D:\\Github\\SLIC\\.env."
                        ) from e
                    time.sleep(1.5 * (attempt + 1))
                except Exception as e:
                    raw = f"ERROR: {e}"
                    time.sleep(1.5 * (attempt + 1))

            results.append({
                "context_role": context_role,
                "formation": formation,
                "condition": condition,
                "seed": seed,
                "expected_dispatch": expected_dispatch,
                "surface_to_role": surface_to_role,
                "dispatched": dispatched,
                "raw": raw[:200],
            })
            pbar.update(1)

    pbar.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} records to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--out", type=str, default="llm_micro/results/dispatches.json")
    args = ap.parse_args()
    run(args.samples, args.model, Path(args.out))


if __name__ == "__main__":
    main()
