###################################
###################################
# EXPERIMENTS: PART 1
# CREATING THE SIMULATIONS DATASET
###################################
###################################

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
import argparse
from tqdm import tqdm
from navground import sim, core
from navground.sim.ui.video import record_video_from_run
import pickle

###################################
# 2: ADJUSTING PARAMETERS
###################################

parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--num_runs', type=int, default=1, help='Number of simulation runs')
parser.add_argument('--num_steps', type=int, default=100, help='Number of steps in the simulation')
parser.add_argument('--time_step', type=float, default=0.1, help='Time step for the simulation')
parser.add_argument('--side', type=float, default=10.0, help='Side of the environment')
parser.add_argument('--num_agents', type=int, default=10, help='Number of agents in the simulation')
parser.add_argument('--radius', type=float, default=0.25, help='Radius of agents')
parser.add_argument('--safety_margin', type=float, default=0.1, help='Safety margin for agents')
parser.add_argument('--max_speed', type=float, default=1.0, help='Maximum speed of agents')
parser.add_argument('--optimal_speed', type=float, default=1.0, help='Optimal speed of agents')
parser.add_argument('--behavior', type=str, default='HL', help='Behavior type')

args = parser.parse_args([
        '--num_runs', '200',
        '--num_steps', '900',
        '--time_step', '0.1',
        '--side', '6.0',
        '--num_agents', '10',
        '--radius', '0.4',
        '--safety_margin', '0.1',
        '--max_speed', '1.66',
        '--optimal_speed', '1.2',
        '--behavior', 'HL'
    ])

###################################
# 3: AUXILIAR FUNCTIONS
###################################

def fill_poses(poses, num_steps):
    steps_to_fill = num_steps - poses.shape[0]
    if steps_to_fill <= 0:
        return poses
    last_pose = poses[-1:]
    fill = np.tile(last_pose, (steps_to_fill, 1, 1))
    return np.concatenate([poses, fill], axis=0)

def fill_efficacies(efficacies, num_steps):
    steps_to_fill = num_steps - efficacies.shape[0]
    if steps_to_fill <= 0:
        return efficacies
    fill = np.zeros((steps_to_fill, efficacies.shape[1]))
    return np.concatenate([efficacies, fill], axis=0)

def fill_safeties(safeties, num_steps):
    steps_to_fill = num_steps - safeties.shape[0]
    if steps_to_fill <= 0:
        return safeties
    fill = np.zeros((steps_to_fill, safeties.shape[1]))
    return np.concatenate([safeties, fill], axis=0)


###################################
# 4: RUNNING THE SIMULATIONS
###################################

behavior_types = ['HL','ORCA','SocialForce']
poses = np.empty((0, args.num_steps, args.num_agents, 3))
efficacies = np.empty((0, args.num_steps, args.num_agents))
safeties = np.empty((0, args.num_steps, args.num_agents))
types = np.empty((0, ), dtype=int)
probes_dict = {key: None for key in ['Efficacies','Safeties','Poses','Types']}

for num_run in tqdm(range(args.num_runs), desc="Progress: ", leave=False):
    for i, btype in enumerate(behavior_types):
        args.behavior = btype
        yaml = f"""
        runs: 1
        steps: {args.num_steps}
        time_step: {args.time_step}
        save_directory: ''
        record_pose: true
        record_twist: true
        record_collisions: true
        record_deadlocks: true
        record_safety_violation: true
        record_efficacy: true
        scenario:
          type: CrossTorus
          side: {args.side}
          groups:
            -
              type: thymio
              number: {args.num_agents}
              radius: {args.radius}
              control_period: 0.1
              speed_tolerance: 0.02
              kinematics:
                type: 2WDiff
                wheel_axis: 0.094
                max_speed: {args.max_speed}
              behavior:
                type: {args.behavior}
                optimal_speed: {args.optimal_speed}
                horizon: 5.0
                safety_margin: {args.safety_margin}
              state_estimation:
                type: Bounded
                range: 5.0
        """
        experiment = sim.load_experiment(yaml)
        experiment.run(start_index=num_run)
        run = experiment.runs[num_run]
        types = np.append(types, i)
        current_poses = np.nan_to_num(np.array([fill_poses(run.poses.copy(),args.num_steps)]),0)
        poses = np.concatenate([poses, current_poses], axis=0)
        current_efficacies = np.nan_to_num(np.array([fill_efficacies(run.efficacy.copy(),args.num_steps)]),0)
        efficacies = np.concatenate([efficacies, current_efficacies], axis=0)
        current_safeties = np.nan_to_num(np.array([fill_safeties(run.safety_violations.copy(),args.num_steps)]),0)
        safeties = np.concatenate([safeties, current_safeties], axis=0)
        if num_run == 0:
            record_video_from_run(run=run, path=f'plots/simulation_{btype}.gif',factor=10.0, fps=30)
    probes_dict = {
        'Efficacies': efficacies,
        'Safeties': safeties,
        'Poses': poses,
        'Types': types}
    with open('files/probes_dict.pkl', 'wb') as file:
        pickle.dump(probes_dict, file) 
