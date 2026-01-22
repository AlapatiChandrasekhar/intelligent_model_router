# ==========================================
# BACKEND LAYER (Business Logic)
#
# Responsibilities:
# - Prompt + file context handling
# - Prompt compression (utility LLM)
# - Complexity detection (utility LLM)
# - Intelligent routing
# - Model invocation (Claude only)
# - Metric calculation
#
# CONSTRAINTS (STRICT):
# - NO stats.json handling
# - Frontend owns persistence
# - Return schema MUST match frontend exactly
# - NO GPT usage
# ==========================================

import os
import time
import warnings
from typing import List, Dict

from google import genai
from anthropic import AnthropicVertex

# ------------------------------------------
# Silence noisy ADC warnings (demo-safe)
# ------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials"
)

# =====================================================
# Model Configuration (NO GPT)
# =====================================================

UTILITY_MODEL = "gemini-3-flash-preview"

SMALL_MODEL = "claude-haiku-4-5"
MEDIUM_MODEL = "claude-sonnet-4-5"
LARGE_MODEL = "claude-opus-4-5"

BASELINE_MODEL = LARGE_MODEL  # used only for relative metrics

# =====================================================
# Simulated cost / latency / carbon values
# (relative, hackathon-safe)
# =====================================================

MODEL_COST_PER_1K_TOKENS = {
    UTILITY_MODEL: 0.005,
    SMALL_MODEL: 0.01,
    MEDIUM_MODEL: 0.03,
    LARGE_MODEL: 0.06,
}

MODEL_LATENCY_SECONDS = {
    UTILITY_MODEL: 0.4,
    SMALL_MODEL: 1.2,
    MEDIUM_MODEL: 2.2,
    LARGE_MODEL: 3.5,
}

CARBON_PER_1K_TOKENS = 0.02  # kgCO2e (simulated)

# =====================================================
# Helper utilities
# =====================================================

def estimate_tokens(text: str) -> int:
    """Approximate token count (sufficient for relative savings)."""
    return max(1, int(len(text.split()) * 1.3))


def extract_file_text(uploaded_files: List) -> str:
    """Extract text from uploaded files (best effort)."""
    content = ""
    for f in uploaded_files or []:
        try:
            content += f.read().decode("utf-8") + "\n"
        except Exception:
            pass
    return content.strip()


def build_full_prompt(user_prompt: str, file_text: str) -> str:
    """Clear separation of user intent and file context."""
    if not file_text:
        return user_prompt.strip()

    return f"""
USER REQUEST:
{user_prompt}

FILE CONTEXT:
{file_text}

INSTRUCTIONS:
- Use file context only if relevant.
- Do not hallucinate missing information.
- Answer ALL requested points.
- Use clear section headers.
""".strip()

# =====================================================
# Retry wrapper (handles Vertex AI 429s)
# =====================================================

def safe_call(fn, *args, retries=2, delay=1):
    for attempt in range(retries + 1):
        try:
            return fn(*args)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < retries:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise

# =====================================================
# Utility LLM (Gemini Flash)
# SAME model for:
# 1) Prompt compression
# 2) Complexity detection
# =====================================================

def compress_prompt(prompt: str) -> str:
    def _call(p):
        client = genai.Client()
        resp = client.models.generate_content(
            model=UTILITY_MODEL,
            contents=f"""
Compress the following input WITHOUT changing meaning.
Remove filler words.
Preserve intent and structure.

Return ONLY compressed text.

Input:
\"\"\"{p}\"\"\"
"""
        )
        return resp.text.strip()

    return safe_call(_call, prompt)


def detect_complexity(prompt: str) -> str:
    def _call(p):
        client = genai.Client()
        resp = client.models.generate_content(
            model=UTILITY_MODEL,
            contents=f"""
Classify the complexity as: easy, medium, or hard.
Return ONLY one word.

Prompt:
\"\"\"{p}\"\"\"
"""
        )
        label = resp.text.strip().lower()
        return label if label in {"easy", "medium", "hard"} else "medium"

    try:
        return safe_call(_call, prompt)
    except Exception:
        # Heuristic fallback (never fail)
        words = len(prompt.split())
        if words < 20:
            return "easy"
        if "design" in prompt.lower() or "architecture" in prompt.lower():
            return "hard"
        return "medium"

# =====================================================
# Routing logic (Claude only)
# =====================================================

def select_model(complexity: str) -> str:
    if complexity == "easy":
        return SMALL_MODEL
    if complexity == "hard":
        return LARGE_MODEL
    return MEDIUM_MODEL

# =====================================================
# Claude invocation (Vertex AI)
# =====================================================

def call_claude(prompt: str, model: str) -> str:
    client = AnthropicVertex(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        region=os.getenv("GOOGLE_CLOUD_LOCATION"),
    )

    max_tokens = 1200 if model == LARGE_MODEL else 700

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

# =====================================================
# PUBLIC API — CALLED BY STREAMLIT
# =====================================================

def get_routing_results(user_prompt: str, uploaded_files: list = None) -> Dict:
    """
    STRICT FRONTEND CONTRACT.
    MUST return exactly:
    {
      "answer": str,
      "model_name": str,
      "cost_saved": float,
      "time_saved": float,
      "carbon_footprint": float,
      "tokens_used": int
    }
    """

    # A) Build full prompt
    file_text = extract_file_text(uploaded_files)
    full_prompt = build_full_prompt(user_prompt, file_text)

    original_tokens = estimate_tokens(full_prompt)

    # B) Compress prompt (utility LLM)
    compressed_prompt = compress_prompt(full_prompt)
    compressed_tokens = estimate_tokens(compressed_prompt)

    # C) Detect complexity (utility LLM)
    complexity = detect_complexity(compressed_prompt)

    # D) Route to best Claude model
    chosen_model = select_model(complexity)

    # E) Generate final answer
    answer = call_claude(compressed_prompt, chosen_model)

    # F) Metric calculation (relative, simulated)
    baseline_cost = (
        original_tokens / 1000
    ) * MODEL_COST_PER_1K_TOKENS[BASELINE_MODEL]

    actual_cost = (
        compressed_tokens / 1000
    ) * MODEL_COST_PER_1K_TOKENS[chosen_model]

    cost_saved = max(baseline_cost - actual_cost, 0.0)

    time_saved = max(
        MODEL_LATENCY_SECONDS[BASELINE_MODEL]
        - MODEL_LATENCY_SECONDS[chosen_model],
        0.0,
    )

    carbon_saved = max(
        (original_tokens - compressed_tokens) / 1000 * CARBON_PER_1K_TOKENS,
        0.0,
    )

    # G) STRICT RETURN (frontend-owned stats)
    return {
        "answer": answer,
        "model_name": chosen_model,
        "cost_saved": round(cost_saved, 4),
        "time_saved": round(time_saved, 3),
        "carbon_footprint": round(carbon_saved, 6),
        "tokens_used": compressed_tokens,
    }
