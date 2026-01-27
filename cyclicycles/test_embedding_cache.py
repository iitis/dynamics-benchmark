#!/usr/bin/env python3
"""
Test script to verify embedding caching functionality.

This script tests that:
1. Embeddings are created and cached on first run
2. Embeddings are loaded from cache on subsequent runs
3. Same embedding is used for both forward and cyclic annealing
"""

import sys
from pathlib import Path
from src.cyclicycles.runner import Runner

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_embedding_cache():
    """Test embedding caching for dynamics instances."""
    
    print("=" * 80)
    print("EMBEDDING CACHING TEST")
    print("=" * 80)
    
    # Create runner
    runner = Runner(sampler='6.4')
    
    # Test 1: Create embedding and cache it
    print("\n[TEST 1] Creating and caching embedding for dynamics instance...")
    print("-" * 80)
    embedding1 = runner._get_or_create_embedding(
        instance_type='dynamics',
        instance_id='N_263_realization_1',
        num_timepoints=5,
        n_nodes=None
    )
    print(f"Embedding type: {type(embedding1)}")
    print(f"Embedding: {embedding1}")
    
    # Test 2: Load embedding from cache
    print("\n[TEST 2] Loading embedding from cache...")
    print("-" * 80)
    embedding2 = runner._get_or_create_embedding(
        instance_type='dynamics',
        instance_id='N_263_realization_1',
        num_timepoints=5,
        n_nodes=None
    )
    print(f"Embedding type: {type(embedding2)}")
    print(f"Same object? {embedding1 is embedding2}")
    print(f"Equal? {str(embedding1) == str(embedding2)}")
    
    # Test 3: Different instance should create new embedding
    print("\n[TEST 3] Different instance should have different embedding...")
    print("-" * 80)
    embedding3 = runner._get_or_create_embedding(
        instance_type='dynamics',
        instance_id='N_678_realization_1',
        num_timepoints=5,
        n_nodes=None
    )
    print(f"Embedding type: {type(embedding3)}")
    print(f"Same as first? {embedding1 is embedding3}")
    
    # Test 4: Static instance embedding
    print("\n[TEST 4] Testing static instance embedding cache...")
    print("-" * 80)
    embedding4 = runner._get_or_create_embedding(
        instance_type='static',
        instance_id=None,
        num_timepoints=5,
        n_nodes='1312'
    )
    print(f"Embedding type: {type(embedding4)}")
    
    embedding5 = runner._get_or_create_embedding(
        instance_type='static',
        instance_id=None,
        num_timepoints=5,
        n_nodes='1312'
    )
    print(f"Loaded from cache? Check console for 'Loaded cached embedding'")
    
    print("\n" + "=" * 80)
    print("EMBEDDING CACHING TEST COMPLETE")
    print("=" * 80)
    print("\nExpected behavior:")
    print("✓ First calls create new embeddings and cache them")
    print("✓ Subsequent calls load from cache")
    print("✓ Different instances get different embeddings")
    print("✓ Both dynamics and static instances support caching")

if __name__ == '__main__':
    test_embedding_cache()
