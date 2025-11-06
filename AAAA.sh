#!/usr/bin/env bash
set -euo pipefail

# Config
PY="${PY:-python3}"            # override with PY=python if needed
SCRIPT="${SCRIPT:-nonogram_woc_ga.py}"  # your main script filename here

# Make test data dir and JSON
mkdir -p tests results
JSON="tests/heart5x5.json"
cat > "$JSON" <<'JSON_EOF'
[
  [0,1,1,1,0],
  [1,0,1,0,1],
  [1,1,1,1,1],
  [0,1,1,1,0],
  [0,0,1,0,0]
]
JSON_EOF

# Helper to run a case in a fresh outdir
run_case () {
  local name="$1"; shift
  local outdir="results/${name}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$outdir"
  echo "==> Running: $name"
  set -x
  $PY "$SCRIPT" --non "$JSON" --outdir "$outdir" "$@"
  set +x
  echo "==> Outputs in: $outdir"
  ls -1 "$outdir" || true
  echo
}

# 0) Quick sanity: show version/help if you added those flags (won't fail if absent)
# $PY "$SCRIPT" -h || true

# 1) Scalar-arg run (trend + gif)
run_case "scalar_args" \
  --pop 800 \
  --gens 600 \
  --elite 0.1 \
  --crowd_ratio 0.25 \
  --tsize 4 \
  --m_parents 4 \
  --p_init 0.5 \
  --snapshot_every 1 \
  --trendplot \
  --animate

# 2) List-arg run (tests parse_*_list + your `_first()` normalization)
#    If you DIDN'T add `_first()`, this should fail — which is the point of the test.
run_case "list_args" \
  --pop 200,300 \
  --gens 100,150 \
  --elite 0.1,0.15 \
  --crowd_ratio 0.2,0.3 \
  --tsize 4,6 \
  --m_parents 3,5 \
  --p_init 0.4,0.6 \
  --snapshot_every 5 \
  --trendplot

# 3) Mixed-case extension check (.JSON)
JSON_UP="tests/HEART5X5.JSON"
cp "$JSON" "$JSON_UP"
run_case "mixed_case_ext" \
  --non "$JSON_UP" \
  --pop 80 \
  --gens 40 \
  --elite 0.1 \
  --crowd_ratio 0.25 \
  --tsize 4 \
  --m_parents 4 \
  --p_init 0.5 \
  --snapshot_every 1 \
  --animate

echo "All test cases completed."

