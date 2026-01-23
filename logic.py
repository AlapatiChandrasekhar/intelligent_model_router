# ==========================================
# 100% LLM-BASED ROUTING - v3.0
# 
# PHILOSOPHY:
# ✓ Every query analyzed by Gemini (no heuristics)
# ✓ Transparent reasoning shown to user
# ✓ Maximum routing accuracy
# ✓ Latency is acceptable trade-off
# ==========================================

import os
import time
import warnings
import json
import base64
import hashlib
from typing import List, Dict, Tuple, Generator
from functools import lru_cache

from google import genai
from anthropic import AnthropicVertex
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials"
)

# =====================================================
# GLOBAL CLIENT INITIALIZATION
# =====================================================
GEMINI_CLIENT = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

CLAUDE_CLIENT = AnthropicVertex(
    project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
    region=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

# =====================================================
# Model Configuration
# =====================================================
UTILITY_MODEL = "gemini-3-flash-preview"
SMALL_MODEL = "claude-haiku-4-5"
MEDIUM_MODEL = "claude-sonnet-4-5"
LARGE_MODEL = "claude-opus-4-5"
BASELINE_MODEL = LARGE_MODEL

# =====================================================
# Cost/Latency/Carbon Configuration
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

MODEL_CARBON_PER_1K_TOKENS = {
    SMALL_MODEL: 0.008,
    MEDIUM_MODEL: 0.015,
    LARGE_MODEL: 0.025,
}

IMAGE_TOKENS_ESTIMATE = 1600

# =====================================================
# Helper Utilities
# =====================================================

def estimate_tokens(text: str, num_images: int = 0) -> int:
    """Improved token estimation with image support."""
    if not text:
        base_tokens = 0
    else:
        words = text.split()
        code_blocks = text.count("```")
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        base_tokens = int(
            len(words) * 1.3 +
            code_blocks * 50 +
            special_chars * 0.1
        )
    
    image_tokens = num_images * IMAGE_TOKENS_ESTIMATE
    return max(1, base_tokens + image_tokens)


def hash_prompt(prompt: str) -> str:
    """Create hash for caching prompts."""
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


def process_uploaded_files(uploaded_files: List) -> Tuple[str, List[Dict]]:
    """Extract text and images from uploaded files."""
    if not uploaded_files:
        return "", []
    
    text_content = ""
    images = []

    for f in uploaded_files:
        try:
            file_type = getattr(f, "type", "")
            
            if file_type.startswith("image/"):
                bytes_data = f.read()
                b64_data = base64.b64encode(bytes_data).decode("utf-8")
                images.append({
                    "media_type": file_type,
                    "data": b64_data
                })
            else:
                text_content += f.read().decode("utf-8", errors="ignore") + "\n"
        except Exception:
            pass
            
    return text_content.strip(), images


def build_full_prompt(user_prompt: str, file_text: str) -> str:
    """Build prompt with clear separation of context."""
    if not file_text:
        return user_prompt.strip()

    return f"""USER REQUEST:
{user_prompt}

FILE CONTEXT:
{file_text}

INSTRUCTIONS:
- Answer the user's request completely and accurately
- Use file context only when relevant
- Provide clear, well-structured responses""".strip()


def safe_call(fn, *args, retries=2, delay=1):
    """Retry wrapper for API calls with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return fn(*args)
        except Exception as e:
            error_str = str(e)
            is_retryable = (
                "429" in error_str or 
                "RESOURCE_EXHAUSTED" in error_str or
                "503" in error_str or
                "timeout" in error_str.lower()
            )
            
            if is_retryable and attempt < retries:
                sleep_time = delay * (2 ** attempt)
                time.sleep(sleep_time)
                continue
            raise


# =====================================================
# CORE: 100% LLM-Based Routing (No Heuristics)
# =====================================================

@lru_cache(maxsize=256)
def analyze_and_route(prompt_hash: str, prompt: str, has_images: bool, has_files: bool) -> Tuple[str, str]:
    """
    100% LLM-based routing decision with transparent reasoning.
    
    Args:
        prompt_hash: Hash for caching
        prompt: Full user prompt
        has_images: Whether images are attached
        has_files: Whether files are attached
    
    Returns:
        (complexity, reasoning) tuple
    """
    
    # Build context hints for the routing LLM
    context_hints = []
    if has_images:
        context_hints.append("User uploaded image(s)")
    if has_files:
        context_hints.append("User uploaded file(s)")
    
    context_str = " | ".join(context_hints) if context_hints else "Text-only query"
    
    routing_prompt = f"""You are the Chief AI Architect responsible for model routing optimization.

GOAL: Maximizing response quality while minimizing inference costs and latency.
Your decisions must be defensible from both an engineering and business perspective.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL CAPABILITY MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 HAIKU (Tier 1: Speed & Efficiency)
   • Use for: "Solved problems" where the answer is known or templated.
   • Cost Efficiency: 6x cheaper than Opus.
   • Ideal for: Greetings, formatting, exact text extraction, simple code (print/loops), translations.
   • "Manager Logic": If Haiku can do it with 98% accuracy, using anything else is waste.

🔸 SONNET (Tier 2: The Generalist)
   • Use for: "Knowledge retrieval" and "Reasoning".
   • Sweet Spot: best balance of intelligence and speed.
   • Ideal for: Explaining concepts, summarizing documents, factual lookups (CEO/Stock), comparative analysis, writing content.
   • "Manager Logic": Use when the user asks "Why?" or "How?", or needs external world knowledge.

🔺 OPUS (Tier 3: Deep Intelligence)
   • Use for: "Novel creation" and "Complex Analysis".
   • Cost: Expensive. Use sparingly.
   • Ideal for: System Architecture (LLD/HLD), Security Audits, Math Proofs, Complex Debugging (Race conditions), Nuanced Creative Writing.
   • "Manager Logic": Use only when failure is not an option or the task requires expert-level deduction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARD GUARDRAILS (SAFETY LOCKS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. COMPLIANCE: If query involves Finance, Medical, Legal, or Security → CANNOT use HAIKU.
2. COMPLEXITY: If query asks for System Design (LLD/HLD), Refactoring, Threat Modeling, or Mathematical Proofs → MUST use OPUS.
3. FILES: If query is "summarize/explain this file" → Prefer SONNET (unless deep analysis is required).
4. TIE-BREAKER: 
   - Haiku vs Sonnet? → Choose SONNET (Safety first).
   - Sonnet vs Opus? → Choose SONNET (Cost optimization, unless a Hard Rule triggers Opus).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context Hints: {context_str}

User Query:
\"\"\"
{prompt}
\"\"\"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyze the query difficulty and select the lowest tier model that satisfies the user's intent WITHOUT compromising accuracy.

Return ONLY valid JSON:
{{
    "complexity": "haiku|sonnet|opus",
    "reasoning": "A professional simple and max 2 line justification explaining why this specific tier is sufficient. Mention cost-efficiency or required capability. (e.g., 'Haiku is capable of handling this simple greeting with zero accuracy loss, optimizing cost and latency.')"
}}
"""

    def _call():
        resp = GEMINI_CLIENT.models.generate_content(
            model=UTILITY_MODEL,
            contents=routing_prompt
        )
        
        text_resp = resp.text.strip()
        
        # Clean markdown fences
        if text_resp.startswith("```"):
            try:
                lines = text_resp.split("\n")
                text_resp = "\n".join(lines[1:-1]) if len(lines) > 2 else text_resp
            except Exception:
                pass
        
        # Parse JSON
        try:
            data = json.loads(text_resp.strip())
            complexity = data.get("complexity", "sonnet")
            reasoning = data.get("reasoning", "Router analysis completed")
            
            # Validate complexity
            if complexity not in ["haiku", "sonnet", "opus"]:
                complexity = "sonnet"
                reasoning = f"Invalid routing ({complexity}), defaulted to sonnet"
            
            return complexity, reasoning
            
        except json.JSONDecodeError:
            # Fallback: Try to extract from text
            text_lower = text_resp.lower()
            
            if '"haiku"' in text_resp or "'haiku'" in text_resp:
                return "haiku", "Simple task detected by router"
            elif '"opus"' in text_resp or "'opus'" in text_resp:
                return "opus", "Complex task detected by router"
            else:
                return "sonnet", "General task detected by router"
    
    try:
        return safe_call(_call)
    except Exception as e:
        # Safe fallback on error
        return "sonnet", f"Router error, using balanced model (Sonnet)"


def select_model(complexity: str) -> str:
    """Map complexity to Claude model tier."""
    if complexity == "haiku":
        return SMALL_MODEL
    elif complexity == "opus":
        return LARGE_MODEL
    else:
        return MEDIUM_MODEL


# =====================================================
# Claude Invocation (Streaming + Non-Streaming)
# =====================================================

def call_claude_stream(prompt: str, model: str, images: List[Dict] = None) -> Generator[str, None, None]:
    """Call Claude API with streaming support."""
    max_tokens = 4096
    
    content_blocks = []
    
    if images:
        for img in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["data"],
                }
            })
    
    content_blocks.append({
        "type": "text",
        "text": prompt
    })
    
    def _api_call():
        return CLAUDE_CLIENT.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content_blocks}],
        )
    
    with safe_call(_api_call) as stream:
        for text_chunk in stream.text_stream:
            yield text_chunk


def call_claude(prompt: str, model: str, images: List[Dict] = None) -> str:
    """Non-streaming Claude call."""
    max_tokens = 4096
    
    content_blocks = []
    
    if images:
        for img in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["data"],
                }
            })
    
    content_blocks.append({
        "type": "text",
        "text": prompt
    })
    
    def _api_call():
        return CLAUDE_CLIENT.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content_blocks}],
        )
    
    msg = safe_call(_api_call)
    return msg.content[0].text


# =====================================================
# PUBLIC API - STREAMING VERSION
# =====================================================

def get_routing_results_stream(user_prompt: str, uploaded_files: list = None) -> Generator:
    """
    100% LLM-routed streaming response.
    
    Yields:
        1. Metadata dict (model_name, cost_saved, reasoning)
        2. Text chunks from Claude
    """
    
    # A) Process files
    file_text, images = process_uploaded_files(uploaded_files)
    num_images = len(images)
    
    # B) Build full prompt
    full_prompt = build_full_prompt(user_prompt, file_text)
    
    # C) 100% LLM ROUTING (No heuristics!)
    prompt_hash = hash_prompt(full_prompt)
    complexity, reasoning = analyze_and_route(
        prompt_hash, 
        full_prompt,
        has_images=(num_images > 0),
        has_files=bool(file_text)
    )
    
    # D) Select model
    chosen_model = select_model(complexity)
    
    # E) Calculate metrics
    tokens = estimate_tokens(full_prompt, num_images=num_images)
    
    baseline_cost = (tokens / 1000) * MODEL_COST_PER_1K_TOKENS[BASELINE_MODEL]
    actual_cost = (tokens / 1000) * MODEL_COST_PER_1K_TOKENS[chosen_model]
    cost_saved = max(0.0, baseline_cost - actual_cost)
    
    time_saved = max(0.0, 
        MODEL_LATENCY_SECONDS[BASELINE_MODEL] - MODEL_LATENCY_SECONDS[chosen_model]
    )
    
    baseline_carbon = (tokens / 1000) * MODEL_CARBON_PER_1K_TOKENS[BASELINE_MODEL]
    actual_carbon = (tokens / 1000) * MODEL_CARBON_PER_1K_TOKENS[chosen_model]
    carbon_saved = max(0.0, baseline_carbon - actual_carbon)
    
    # F) Yield metadata with reasoning
    metadata = {
        "model_name": chosen_model,
        "cost_saved": round(cost_saved, 4),
        "time_saved": round(time_saved, 3),
        "carbon_footprint": round(carbon_saved, 6),
        "routing_reason": reasoning
    }
    
    yield metadata
    
    # G) Stream Claude response
    for chunk in call_claude_stream(full_prompt, chosen_model, images=images):
        yield chunk


# =====================================================
# PUBLIC API - NON-STREAMING
# =====================================================

def get_routing_results(user_prompt: str, uploaded_files: list = None) -> Dict:
    """Non-streaming version with 100% LLM routing."""
    
    file_text, images = process_uploaded_files(uploaded_files)
    num_images = len(images)
    
    full_prompt = build_full_prompt(user_prompt, file_text)
    
    # 100% LLM routing
    prompt_hash = hash_prompt(full_prompt)
    complexity, reasoning = analyze_and_route(
        prompt_hash,
        full_prompt,
        has_images=(num_images > 0),
        has_files=bool(file_text)
    )
    
    chosen_model = select_model(complexity)
    
    answer = call_claude(full_prompt, chosen_model, images=images)
    
    tokens = estimate_tokens(full_prompt, num_images=num_images)
    
    baseline_cost = (tokens / 1000) * MODEL_COST_PER_1K_TOKENS[BASELINE_MODEL]
    actual_cost = (tokens / 1000) * MODEL_COST_PER_1K_TOKENS[chosen_model]
    cost_saved = max(0.0, baseline_cost - actual_cost)
    
    time_saved = max(0.0, 
        MODEL_LATENCY_SECONDS[BASELINE_MODEL] - MODEL_LATENCY_SECONDS[chosen_model]
    )
    
    baseline_carbon = (tokens / 1000) * MODEL_CARBON_PER_1K_TOKENS[BASELINE_MODEL]
    actual_carbon = (tokens / 1000) * MODEL_CARBON_PER_1K_TOKENS[chosen_model]
    carbon_saved = max(0.0, baseline_carbon - actual_carbon)
    
    return {
        "answer": answer,
        "model_name": chosen_model,
        "cost_saved": round(cost_saved, 4),
        "time_saved": round(time_saved, 3),
        "carbon_footprint": round(carbon_saved, 6),
        "routing_reason": reasoning
    }