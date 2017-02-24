
parallel --ungroup python fit_potentials.py --no-cache ${1} --num-jobs 4 --job-id ::: 0 1 2 3

