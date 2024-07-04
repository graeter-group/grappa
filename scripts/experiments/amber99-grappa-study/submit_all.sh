SCRIPTDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# cd to root:
pushd "${SCRIPTDIR}/../../../" > /dev/null

echo "Running from $(pwd)"

# Submit all jobs:
declare -a NAMES=("all" "torsion" "angle" "bond" "angle-torsion" "bond-angle" "bond-torsion")
declare -a REF_TERMS=("[nonbonded]" "[nonbonded,bond,angle]" "[nonbonded,bond,proper,improper]" "[nonbonded,angle,proper,improper]" "[nonbonded,bond]" "[nonbonded,proper,improper]" "[nonbonded,angle]")
declare -a ENERGIES=("[bond,angle,proper,improper]" "[proper,improper]" "[angle]" "[bond]" "[angle,proper,improper]" "[bond,angle]" "[bonds,proper,improper]")

for i in "${!NAMES[@]}"; do
    # if [ $i -eq 1 ]; then
    #     break
    # fi
    NAME=${NAMES[i]}
    REF_TERM=${REF_TERMS[i]}
    ENERGY=${ENERGIES[i]}

    jobstr="sbatch job.sh python experiments/train.py model=default experiment.wandb.project=grappa-amber-study experiment.wandb.name=$NAME data=amber99-torsion data.data_module.ref_terms=$REF_TERM data.energy.terms=$ENERGY"
    # echo "$jobstr"
    $jobstr
done

# cd back:
popd > /dev/null