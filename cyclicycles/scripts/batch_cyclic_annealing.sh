#!/bin/bash

# Batch runner script for cyclic annealing on dynamics instances
# Runs cyclic annealing for all instances (1-8) and timepoints (2, 6, 7, 8, 9, 10, 11, 12)
# Each run executes exactly one cyclic annealing sample and exits

TIMEPOINTS=(3 4 5 6 7 8 9 10 11 12)
INSTANCES=(1 2 3 4 5 6 7 8)
NUM_READS=1000
NUM_CYCLES=5
SAMPLES_PER_CONFIG=3

echo "=============================================================================="
echo "Batch Cyclic Annealing Runner"
echo "=============================================================================="
echo "Parameters:"
echo "  - Instances: ${INSTANCES[@]}"
echo "  - Timepoints: ${TIMEPOINTS[@]}"
echo "  - Reads per cycle: $NUM_READS"
echo "  - Cycles: $NUM_CYCLES"
echo "  - Samples per configuration: $SAMPLES_PER_CONFIG"
echo "=============================================================================="

TOTAL_CONFIGS=$((${#INSTANCES[@]} * ${#TIMEPOINTS[@]} * SAMPLES_PER_CONFIG))
echo ""
echo "Total configurations to run: $TOTAL_CONFIGS"
echo ""

SUCCESSFUL=0
FAILED=0
CURRENT=0

# Iterate through all timepoints first, then instances
for TIMEPOINT in "${TIMEPOINTS[@]}"; do
    echo ""
    echo "[Timepoints $TIMEPOINT]"
    echo "--------------------------------------------------------------------------"
    
    for INSTANCE in "${INSTANCES[@]}"; do
        for ((SAMPLE=1; SAMPLE<=SAMPLES_PER_CONFIG; SAMPLE++)); do
            CURRENT=$((CURRENT + 1))
            
            echo "  [$CURRENT/$TOTAL_CONFIGS] Running cyclic annealing (instance=$INSTANCE, timepoints=$TIMEPOINT, sample=$SAMPLE/$SAMPLES_PER_CONFIG, reads=$NUM_READS, cycles=$NUM_CYCLES)..."
            
            # Run the Python script and capture the exit code
            python scripts/cyclic_annealing.py \
                --sampler "6.4" \
                --instance-id "$INSTANCE" \
                --num-timepoints "$TIMEPOINT" \
                --num-reads "$NUM_READS" \
                --num-cycles "$NUM_CYCLES"
            
            EXIT_CODE=$?
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "    ✓ Success"
                SUCCESSFUL=$((SUCCESSFUL + 1))
            else
                echo "    ✗ Failed (exit code: $EXIT_CODE)"
                FAILED=$((FAILED + 1))
            fi
            
            # Small delay between runs
            sleep 1
        done
    done
done

# Summary
echo ""
echo "=============================================================================="
echo "SUMMARY"
echo "=============================================================================="
echo "Successful runs: $SUCCESSFUL/$TOTAL_CONFIGS"
echo "Failed runs: $FAILED/$TOTAL_CONFIGS"
if [ $TOTAL_CONFIGS -gt 0 ]; then
    SUCCESS_RATE=$((100 * SUCCESSFUL / TOTAL_CONFIGS))
    echo "Success rate: $SUCCESS_RATE%"
fi
echo "=============================================================================="
