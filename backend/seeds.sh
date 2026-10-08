for s in 1 2 3; do
  mkdir -p artifacts_seed$s
  python3 train.py --seed $s --variants mode_aware --out artifacts_seed$s > seed$s.log 2>&1
done
echo ALLDONE >> seed3.log
