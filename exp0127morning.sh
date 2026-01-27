#!/usr/bin/env bash
set -u
set -o pipefail

BIN="./bin/main"

# --- 実行前チェック（ここでOK） ---
command -v timeout >/dev/null || { echo "timeout not found"; exit 1; }
[ -x "$BIN" ] || { echo "binary not executable: $BIN"; exit 1; }

OUTDIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

DATASETS=(
  "sift100k.txt"
  "sift1m.txt"
  "mnist_train.txt"
  "gist1m.txt"
)
DATASET_ADDITIONAL=(
  "sift10m.txt"
)

KS=(20)
RHO="0.5"
REPEAT=10

# 1回あたりの上限時間（必要に応じて調整）
TIME_LIMIT="2h"

BASELINE_OPTS=()
NND_OPTS=(--full-NND)

PROPOSED_1ITER=(--lsh --prefilter --proj-P 8 --lsh-L 4 --lsh-K 2 --lsh-w 64 --lsh-bucket 64 --lsh-cand 48 --pf-mult 2 --k-cap 30 --keep 0.5 --refine-iter 1)
PROPOSED_2ITER=(--lsh --prefilter --proj-P 8 --lsh-L 4 --lsh-K 2 --lsh-w 64 --lsh-bucket 64 --lsh-cand 48 --pf-mult 2 --k-cap 30 --keep 0.5 --refine-iter 2)
PROPOSED_3ITER=(--lsh --prefilter --proj-P 8 --lsh-L 4 --lsh-K 2 --lsh-w 64 --lsh-bucket 64 --lsh-cand 48 --pf-mult 2 --k-cap 30 --keep 0.5 --refine-iter 3)
PROPOSED_OPTS_OPTB=(--lsh --prefilter --proj-P 8 --lsh-L 4 --lsh-K 2 --lsh-w 64 --lsh-bucket 64 --lsh-cand 48 --pf-mult 2 --k-cap 30 --keep 0.5)

FAIL_LOG="${OUTDIR}/failures.tsv"
printf "when\ttag\tdataset\tk\trun\texit_code\n" > "$FAIL_LOG"

run_one () {
  local dataset="$1"
  local k="$2"
  local tag="$3"
  shift 3
  local opts=("$@")

  local log="${OUTDIR}/${tag}__${dataset}__k${k}.log"
  echo "### ${tag} dataset=${dataset} k=${k} $(date)" | tee -a "$log"

  for r in $(seq 1 $REPEAT); do
    echo "--- run ${r}/${REPEAT} --- $(date)" | tee -a "$log"

    (
      timeout --signal=SIGTERM --kill-after=30s "$TIME_LIMIT" \
        "$BIN" "$dataset" "$k" "$RHO" "${opts[@]}"
    ) >>"$log" 2>&1

    ec=$?
    if [ $ec -ne 0 ]; then
      # 124: timeout で終了したことが多い
      echo "!!! ERROR exit_code=$ec (continuing)" | tee -a "$log"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$(date +%F_%T)" "$tag" "$dataset" "$k" "$r" "$ec" >> "$FAIL_LOG"
      # 連続失敗で暴走するのが怖ければ少し待つ
      # sleep 2
    fi
  done
}

for ds in "${DATASETS[@]}"; do
  for k in "${KS[@]}"; do
    run_one "$ds" "$k" "k_cap30_1iter" "${PROPOSED_1ITER[@]}"
    run_one "$ds" "$k" "k_cap30_2iter" "${PROPOSED_2ITER[@]}"
    run_one "$ds" "$k" "k_cap30_3iter"    "${PROPOSED_3ITER[@]}"
  done
done

for ds in "${DATASET_ADDITIONAL[@]}"; do
  for k in "${KS[@]}"; do
      run_one "$ds" "$k" "k_cap30_1iter" "${PROPOSED_1ITER[@]}"
      run_one "$ds" "$k" "k_cap30_2iter" "${PROPOSED_2ITER[@]}"
      run_one "$ds" "$k" "k_cap30_3iter"    "${PROPOSED_3ITER[@]}"
    done
done

echo "All done. Logs in $OUTDIR"
echo "Failures (if any): $FAIL_LOG"