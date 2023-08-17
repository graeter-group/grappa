# how many times to execute the tripeptides command:
n=${1:-1}

for ((i=1; i<=$n; i++)); do
    echo "Executing iteration $i"
    sbatch run.sh bash tripeptides.sh
done