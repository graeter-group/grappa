import wandb
import time
import numpy as np
from grappa.training.resume_trainrun import resume_trainrun
import sys

def get_runs(project:str, states=['crashed', 'failed', 'killed']):
    """
    """

    # Initialize the W&B API
    api = wandb.Api()

    # Fetch runs from the specified project
    runs = api.runs(f"{project}")

    rundict = {state:[] for state in states}

    # Iterate over runs and check their status
    for run in runs:
        if run.state in states:
            rundict[run.state].append(run)
    return rundict


def resume_agent(project:str, time_limit:float=23, overwrite_config={}, get_crashed_runs=get_runs):
    """
    For the time specified in time_limit, resume a random runs of those that have a state in states. If None is found, wait for 5 minutes and try again.
    """
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit*60*60:
        # get all runs that are in a crashed or failed state:
        run_dict = get_crashed_runs(project=project)

        counter_dict = {state:len(runs) for state, runs in run_dict.items()}

        print(f"encountered stopped runs:\n{counter_dict}")

        stopped_runs = []

        # sort states by priority:
        for state in ['crashed', 'failed', 'killed', 'finished']:
            if state in run_dict.keys():
                if len(run_dict[state]) > 0:
                    stopped_runs = run_dict[state]
                    break

        if len(stopped_runs) == 0:
            print("No stopped runs found. Waiting for 5 minutes...")
            time.sleep(5*60)
            continue

        # choose a random run:
        run_id = stopped_runs[np.random.randint(len(stopped_runs))].id

        print(f"Resuming run {run_id}...", file=sys.stderr) # make it appear in the stderr stream to be found more conventiently
        print(f"Resuming run {run_id}...")

        # resume it:
        try:
            resume_trainrun(run_id=run_id, project=project, new_wandb_run=False, overwrite_config=overwrite_config)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
        except:
            print(f"Resuming run {run_id} failed.")
            print(f"Resuming run {run_id} failed.", file=sys.stderr)
            continue
        print(f"Resuming run {run_id} succeeded.")
        print(f"Resuming run {run_id} succeeded.", file=sys.stderr)


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--continue_finished', action='store_true', help='continue all finished runs with runtime > 10 hours')

    args = parser.parse_args()

    if not args.continue_finished:
        resume_agent(project='hpo_grappa_final', time_limit=23, overwrite_config={'data_config': {'val_batch_size':64}}) # resume all crashed runs


    else:
        def continue_good_runs(project):
            """
            Filter for finished runs with runtime longer than 10 hours.
            """
            api = wandb.Api()
            runs = api.runs(f"{project}")

            out_runs = []
            for run in runs:
                if run.state == 'finished':
                    if run.summary['_wandb']['runtime'] > 10 * 60 * 60:
                        out_runs.append(run)
            return {'finished':out_runs}


        resume_agent(project='hpo_grappa_final', time_limit=23, overwrite_config={'lit_model_config':{'time_limit':30, "finish_criterion": {1:50, 2:30, 4:20, 10:17, 15:16, 24:15.5}}}, get_crashed_runs=continue_good_runs) # continue runs that were terminated after 15 hours due to the old time limit of 15 hours

