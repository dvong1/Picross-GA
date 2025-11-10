#!/bin/bash

OUTDIR="non_ga_results"
mkdir -p "$OUTDIR"

# can modify these parameters as needed for further experimenting
POPS=(100 200)
GENS=(200 400)
ELITE=0.1
CROWD=0.2
TSIZE=4
MPARENTS=4
SEED=42

PUZZLES=("nons/letterP.non" "nons/spade.non" "nons/mouse.non" "nons/UofL.non")

for PUZ in "${PUZZLES[@]}"; do
  for POP in "${POPS[@]}"; do
    for GEN in "${GENS[@]}"; do
      echo "===================================================="
      echo "Running puzzle: $PUZ | pop=$POP | gens=$GEN"
      echo "===================================================="

      python3 nonogram_woc_ga.py \
        --non "$PUZ" \
        --pop "$POP" \
        --gens "$GEN" \
        --elite "$ELITE" \
        --crowd_ratio "$CROWD" \
        --tsize "$TSIZE" \
        --m_parents "$MPARENTS" \
        --seed "$SEED" \
        --trendplot \
        --comparepng \
        --outdir "$OUTDIR"
    done
  done
done
