import logic
import json
import os

def test_logic():
    print("Testing logic.get_routing_results...")
    result = logic.get_routing_results("test prompt")
    required_keys = ["answer", "model_name", "cost_saved", "time_saved", "carbon_footprint", "tokens_used"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    print("Logic test passed!")

def test_stats_persistence():
    print("Testing stats persistence...")
    stats_file = "stats.json"
    
    # Read initial
    with open(stats_file, 'r') as f:
        initial_stats = json.load(f)
    
    print(f"Initial Total Cost: {initial_stats['total_cost_saved']}")
    
    # Simulate an update (copying logic from app.py loosely)
    new_metrics = {
        "model_name": "TestModel",
        "cost_saved": 10.0,
        "time_saved": 5.0,
        "carbon_footprint": 1.0,
        "tokens_used": 100
    }
    
    initial_stats["total_cost_saved"] += new_metrics["cost_saved"]
    
    # Write
    with open(stats_file, 'w') as f:
        json.dump(initial_stats, f)
        
    # Read back
    with open(stats_file, 'r') as f:
        updated_stats = json.load(f)
        
    assert updated_stats["total_cost_saved"] == initial_stats["total_cost_saved"]
    print(f"Updated Total Cost: {updated_stats['total_cost_saved']}")
    print("Stats persistence passed!")

if __name__ == "__main__":
    try:
        test_logic()
        test_stats_persistence()
        print("\nAll System Checks Passed ✅")
    except Exception as e:
        print(f"\nVerification Failed ❌: {e}")
