#!/bin/bash


run_test() {
    script=$1
    alg_name=$2

    echo "Testing $alg_name ..."

    deterministic=true

    for i in {1..10}; do
        python3 "$script" --out file1.txt --dtype float32 --seed $i
        python3 "$script" --out file2.txt --dtype float32 --seed $i

        if ! diff -q file1.txt file2.txt >/dev/null 2>&1; then
            deterministic=false
            echo "$alg_name: mismatch at iteration $i"
            break
        fi
    done

    if [ "$deterministic" = true ]; then
        echo "$alg_name is deterministic"
    else
        echo "$alg_name NOT deterministic"
    fi

    rm -f file1.txt file2.txt
}



run_test cupy_alg1.py "alg1"
run_test cupy_alg2.py "alg2"
run_test cupy_alg3.py "alg3"