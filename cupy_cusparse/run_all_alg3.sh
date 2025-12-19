#!/usr/bin/env bash
set -euo pipefail


OUTDIR="${1:-dump_alg3_txt}"
SIZES="${SIZES:-"32 64 128 256 512 1024"}"
DENSITIES="${DENSITIES:-"0.01 0.1 0.3 0.5"}"
CHUNK_FRACTION="${CHUNK_FRACTION:-0.2}"  


PYGEN_PY="${PYGEN_PY:-python3}"
PYGEN_SCRIPT="${PYGEN_SCRIPT:-gen_and_save_alg3_txt.py}"


CUEXE="${CUEXE:-./spgemm_from_txt_alg3}"


CMPPY_PY="${CMPPY_PY:-python3}"
CMPPY_SCRIPT="${CMPPY_SCRIPT:-./compare_csrs_txt.py}"


STRICT="${STRICT:-}"   




echo "==[1/3] generate A/B/C(py, ALG3) to $OUTDIR =="
"$PYGEN_PY" "$PYGEN_SCRIPT" \
  --sizes $SIZES \
  --densities $DENSITIES \
  --outdir "$OUTDIR" \
  --chunk-fraction "$CHUNK_FRACTION"


echo "==[2/3] use cuSPARSE ALG3 for case C(cu) =="
shopt -s nullglob
cases=("$OUTDIR"/A_n*_dens*_alg3_indptr.txt)
if [ ${#cases[@]} -eq 0 ]; then
  echo "can't A_* file(_alg3_), please $OUTDIR content." >&2
  exit 2
fi

pass=0; fail=0; total=0
report="$OUTDIR/report_alg3.txt"
: > "$report"

for a_indptr in "${cases[@]}"; do

  prefix="${a_indptr%_indptr.txt}"
  tag="$(basename "$prefix")"      # A_n{N}_dens{dp}_alg3
  base="${tag#A_}"                 # n{N}_dens{dp}_alg3

  Apre="$OUTDIR/A_${base}"
  Bpre="$OUTDIR/B_${base}"
  Cpy="$OUTDIR/C_py_${base}"
  Ccu="$OUTDIR/C_cu_${base}"


  for need in "${Bpre}_indptr.txt" "${Cpy}_indptr.txt"; do
    if [ ! -f "$need" ]; then
      echo "[SKIP] loss:$need" | tee -a "$report"
      continue 2
    fi
  done

  echo "-> [$base] C++ Calculating..."
  "$CUEXE" "$Apre" "$Bpre" "$Ccu" >/dev/null

  echo "   ComParing..."
  if "$CMPPY_PY" "$CMPPY_SCRIPT" "$Cpy" "$Ccu" $STRICT >/dev/null; then
    echo "[PASS] $base" | tee -a "$report"
    pass=$((pass+1))
  else
    echo "[FAIL] $base" | tee -a "$report"
    fail=$((fail+1))
  fi
  total=$((total+1))
done

echo "==[3/3] finish:$pass PASS / $fail FAIL / $total TOTAL =="
echo "Report:$report"
