#!/usr/bin/env bash
set -euo pipefail


OUTDIR="${1:-dump_alg1_txt}"
SIZES="${SIZES:-"32 64 128 256 512 1024"}"
DENSITIES="${DENSITIES:-"0.01 0.1 0.3 0.5"}"


PYGEN_PY="${PYGEN_PY:-python3}"
PYGEN_SCRIPT="${PYGEN_SCRIPT:-gen_and_save_alg1_txt.py}"


CUEXE="${CUEXE:-./spgemm_from_txt_alg1}"


CMPPY_PY="${CMPPY_PY:-python3}"
CMPPY_SCRIPT="${CMPPY_SCRIPT:-./compare_csrs_txt.py}"


STRICT="${STRICT:-}"




# ===== 1/3 Generate Matrix =====
echo "==[1/3] generate Sparse Matrix A/B/C(py) to $OUTDIR =="
"$PYGEN_PY" "$PYGEN_SCRIPT" --sizes $SIZES --densities $DENSITIES --outdir "$OUTDIR"

# ===== 2/3 C++ Compute and generate output =====
echo "==[2/3] use cuSPARSE ALG1 for case  C(cu) =="
shopt -s nullglob
cases=("$OUTDIR"/A_n*_dens*_alg1_indptr.txt)
if [ ${#cases[@]} -eq 0 ]; then
  echo "Can't find A_* file, please check $OUTDIR content" >&2
  exit 2
fi

pass=0; fail=0; total=0
report="$OUTDIR/report_alg1.txt"
: > "$report"

for a_indptr in "${cases[@]}"; do

  prefix="${a_indptr%_indptr.txt}"
  tag="$(basename "$prefix")"     
  base="${tag#A_}"               

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

  echo "-> [$base] C++ Computing..."
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
