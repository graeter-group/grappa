SCRIPTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# cd to root:
pushd "${SCRIPTDIR}/../../../" > /dev/null

echo "Running from $(pwd)"

# Submit all jobs:
declare -a NAMES=(2 3 5 8 10 15 20 30 40 50)

for i in "${!NAMES[@]}"; do
    # if [ $i -eq 1 ]; then
    #     break
    # fi

    NAME=${NAMES[$i]}

    mkdir -p logs

    jobstr="sbatch job.sh python experiments/train.py model=default experiment.wandb.project=grappa-conf-study experiment.wandb.name=confs-$NAME data=dipeptides data.data_module.tr_max_confs=$NAME"
    # echo "$jobstr"
    $jobstr
done

# cd back:
popd > /dev/null