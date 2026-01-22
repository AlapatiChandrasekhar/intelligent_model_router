import random

def get_routing_results(prompt: str, files: list = None) -> dict:
    """
    Simulates the backend routing logic.
    Returns a dictionary with the answer and calculated metrics.
    """
    files_info = f" with {len(files)} file(s)" if files else ""
    # Simulate processing (Dummy Data)
    return {
        "answer": f"Processed answer for: '{prompt}'{files_info} using the best model.",
        "model_name": random.choice(["GPT-4o", "Claude 3.5 Sonnet", "Llama 3 70B"]),
        "cost_saved": round(random.uniform(0.01, 0.50), 4),  # $
        "time_saved": round(random.uniform(0.1, 2.0), 2),    # seconds
        "carbon_footprint": round(random.uniform(0.0001, 0.005), 5), # kgCO2e
        "tokens_used": random.randint(50, 500)
    }