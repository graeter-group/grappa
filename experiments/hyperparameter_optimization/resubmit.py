"""
Script that runs in the background and submits a given job so often that the total number of joibs requested by a given user is kept at a constant value.
"""

import time
import os
import subprocess

def get_slurm_job_count(username):
    command = f"squeue -u {username} | grep -c '^'"
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        job_count = int(output.strip()) - 1
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        job_count = 0
    return job_count


def submit(cmd, n_total, user):
    # get the number of jobs currently running from the given user:
    n_running = get_slurm_job_count(user)

    print(f"n_running: {n_running}")

    # submit jobs until the total number of jobs is reached:
    for i in range(n_total - n_running):
        os.system(cmd)

    print()


def resubmit_agent(cmd, n_total, user, sleep_time=0.5, runtime=72):
    '''
    Times are in hours.
    '''
    start = time.time()
    
    while (time.time() - start) < runtime*60*60:
        submit(cmd, n_total, user)
        time.sleep(sleep_time*60*60)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', type=str, help='command to execute', default='python submit.py -n 1')
    parser.add_argument('-n', '--n_total', type=int, help='total number of jobs to be running at the same time', default=10)
    parser.add_argument('-u', '--user', type=str, help='user name', default='seutelf')
    parser.add_argument('-s', '--sleep_time', type=float, help='sleep time in seconds', default=0.5)
    parser.add_argument('-r', '--runtime', type=float, help='runtime in hours', default=72)

    args = parser.parse_args()

    resubmit_agent(args.cmd, args.n_total, args.user, args.sleep_time, args.runtime)
