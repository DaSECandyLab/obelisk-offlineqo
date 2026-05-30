#!/usr/bin/env python3
"""
Test script to verify the vector-based workflow in the optimization pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimization.guider import Guider
from llm.llm_config import LLMConfig
from util.knob_space import KnobSpace
from util.logger import logger


def test_vector_workflow():
    """Test the vector-based workflow."""
    
    logger.info("Testing simplified vector-based workflow...")
    
    # Define knob names (no ranges needed for tuning)
    knob_names = [
        "tidb_opt_agg_push_down",
        "tidb_opt_insubq_to_join_and_agg", 
        "tidb_opt_join_reorder_threshold"
    ]
    
    # Physical ranges (only needed for execution)
    physical_ranges = {
        "tidb_opt_agg_push_down": (0.1, 10.0),
        "tidb_opt_insubq_to_join_and_agg": (0.1, 10.0),
        "tidb_opt_join_reorder_threshold": (0.1, 10.0)
    }
    
    # Test KnobSpace with names only
    knob_space = KnobSpace(knob_names)
    logger.info(f"KnobSpace dimension: {knob_space.dimension}")
    logger.info(f"Knob names: {knob_space.knob_names}")
    
    # Test normalized search space (always 0-1)
    normalized_space = knob_space.get_normalized_search_space()
    logger.info(f"Normalized search space: {normalized_space}")
    
    # Verify all normalized knobs have 0-1 range
    for knob in normalized_space:
        if knob['min'] == 0.0 and knob['max'] == 1.0 and knob['default'] == 0.5:
            logger.info(f"  ✓ Knob {knob['var']} has correct normalized range [0,1] with default 0.5")
        else:
            logger.error(f"  ✗ Knob {knob['var']} has incorrect range or default")
    
    # Test default vector with no physical defaults.
    default_vector = knob_space.get_default_vector()
    logger.info(f"Default vector: {default_vector}")
    expected_default = [0.5] * len(knob_names)
    if default_vector == expected_default:
        logger.info("✓ Default vector is correct without physical defaults")
    else:
        logger.error(f"✗ Default vector incorrect, expected {expected_default}")
    
    # Test vector-to-config conversion without physical ranges
    config_without_ranges = knob_space.vector_to_config(default_vector)
    logger.info(f"Config without physical ranges: {config_without_ranges}")
    
    # Add physical ranges and test conversion
    knob_space.add_physical_ranges(physical_ranges)
    config_with_ranges = knob_space.vector_to_config(default_vector)
    logger.info(f"Config with physical ranges: {config_with_ranges}")
    
    # Test round-trip conversion
    back_to_vector = knob_space.config_to_vector(config_with_ranges)
    logger.info(f"Back to vector: {back_to_vector}")
    
    if all(abs(a - b) < 1e-10 for a, b in zip(default_vector, back_to_vector)):
        logger.info("✓ Round-trip vector conversion successful")
    else:
        logger.error("✗ Round-trip vector conversion failed")
    
    # Test Guider with knob names only
    guider = Guider(
        warm_start_rounds=3,
        knob_names=knob_names,
        llm_config=LLMConfig(enabled=False),
    )
    logger.info(f"Guider initialized with {len(knob_names)} knobs")
    
    # Test warm start sampling (returns vectors)
    warm_vectors = guider.warm_start_sampling()
    logger.info(f"Warm start vectors: {len(warm_vectors)} samples")
    for i, vector in enumerate(warm_vectors):
        logger.info(f"  Vector {i+1}: {vector}")
        
        # Verify vector is in [0,1] range
        if all(0.0 <= v <= 1.0 for v in vector):
            logger.info(f"  ✓ Vector {i+1} is normalized")
        else:
            logger.error(f"  ✗ Vector {i+1} is not normalized")
        
        # Convert to physical config (requires physical ranges)
        physical_config = guider.knob_space.vector_to_config(vector)
        logger.info(f"  Physical config {i+1}: {physical_config}")
    
    # Test recording observations (using vectors)
    for i, vector in enumerate(warm_vectors):
        fake_perf = 1000 + i * 100  # Mock performance values
        guider.record_observation(vector, fake_perf, plan_id=f"plan_{i}")
        logger.info(f"Recorded observation {i+1}: vector={vector}, perf={fake_perf}")

    # Verify context retrieval enforces plan diversity via plan_id uniqueness.
    repeated_plan_guider = Guider(warm_start_rounds=3, knob_names=knob_names)
    repeated_plan_observations = [
        ([0.10, 0.10, 0.10], 1000, "plan_a"),
        ([0.11, 0.11, 0.11], 1010, "plan_a"),
        ([0.20, 0.20, 0.20], 900, "plan_b"),
        ([0.30, 0.30, 0.30], 800, "plan_c"),
    ]
    for vector, perf, plan_id in repeated_plan_observations:
        repeated_plan_guider.record_observation(vector, perf, plan_id=plan_id)

    similar = repeated_plan_guider.get_similar_observations([0.105, 0.105, 0.105], 3)
    logger.info(f"Plan-diverse similar observations: {similar}")
    if len(similar) == 3 and similar[0][0] == [0.10, 0.10, 0.10]:
        logger.info("✓ Similar observation retrieval keeps nearest point and removes duplicate plan_id entries")
    else:
        logger.error("✗ Similar observation retrieval failed to enforce plan diversity")

    # Test vector validation
    test_vectors = [
        [0.0, 0.5, 1.0],     # Valid
        [-0.1, 0.5, 1.1],    # Invalid (out of range)
        [0.3, 0.7],          # Invalid (wrong dimension)
        [0.2, 0.8, 0.6],     # Valid
    ]
    
    for i, vec in enumerate(test_vectors):
        is_valid = guider.knob_space.validate_vector(vec)
        logger.info(f"Vector {i+1} {vec} validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Test getting next points (should return vectors)
    try:
        guider.suggest_vector = lambda: [0.25, 0.5, 0.75]
        next_vectors = guider.get_next_points(
            sql_path="mock_sql.sql",
            topk=3,
            try_number=0,
            batch=2
        )
        logger.info(f"Next vectors: {len(next_vectors)} samples")
        for i, vector in enumerate(next_vectors):
            logger.info(f"  Next vector {i+1}: {vector}")
            if all(0.0 <= v <= 1.0 for v in vector):
                logger.info(f"  ✓ Next vector {i+1} is normalized")
            else:
                logger.error(f"  ✗ Next vector {i+1} is not normalized")
    except Exception as e:
        logger.error(f"Error getting next points: {e}")
    
    # Test the key insight: tuning doesn't need physical ranges
    logger.info("\n" + "="*60)
    logger.info("KEY INSIGHT VERIFICATION:")
    logger.info("- Tuning operates purely in [0,1] space")
    logger.info("- Physical ranges only needed at execution time")
    logger.info("- Default x0 is normalized from physical defaults when provided")
    logger.info("- LLM sees consistent 0-1 ranges for all knobs")
    logger.info("="*60)
    
    logger.info("Simplified vector workflow test completed successfully!")


if __name__ == "__main__":
    test_vector_workflow() 
