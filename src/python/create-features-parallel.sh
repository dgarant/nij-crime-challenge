

rm ../../features/count-features-550.csv.gz?
parallel --ungroup python create-features.py --num-jobs 4 --job-id ::: 0 1 2 3
cat ../../features/count-features-550.csv.gz? > ../../features/count-features-550.csv.gz

