###################################
###################################
# EXAMPLE WITH 3 AGENTS: PART 2
# CREATING THE SIGNALS
###################################
###################################

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from gudhi.wasserstein import wasserstein_distance
import argparse
from functools import partial
from tslearn import metrics
import perdiver.perdiver as perdiver

poses = np.load('files/poses_3agents.npy')

###################################
# 2: ADJUSTING PARAMETERS
###################################

parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--time_delay', type=int, default=1, help='Time delay to analise simulation intervals')
parser.add_argument('--embedding_length', type=int, default=10, help='Length of the simulation intervals')

args = parser.parse_args([
        '--time_delay', '1',
        '--embedding_length', '5',
    ])

###################################
# 3: AUXILIAR FUNCTIONS
###################################

def custom_distance(vector1, vector2, weights):
    result = 0
    if weights[0] != 0:
        px_diff = np.abs(vector1[0] - vector2[0])
        result += px_diff * weights[0]
    if weights[1] != 0:
        py_diff = np.abs(vector1[1] - vector2[1])
        result += py_diff * weights[1]
    if weights[2] != 0:
        pr_diff = np.abs(vector1[2] - vector2[2])
        pr_diff = np.minimum(pr_diff, 2 * np.pi - pr_diff)
        result += pr_diff * weights[2]
    return result

weights = np.array([1,1/5,1/np.pi])
custom_distance_with_param = partial(custom_distance, weights=weights)

def dismat_from_steps(trajectories, steps):
    num_agents = trajectories.shape[1]
    dismat = np.zeros((num_agents, num_agents))
    for a in range(num_agents):
        for b in range(a+1):
            _, tsim = metrics.dtw_path_from_metric(trajectories[steps,a,:], trajectories[steps,b,:], metric=custom_distance_with_param)
            dismat[a,b] = tsim
    return dismat

def compute_dismat_list(trajectories, args):
    sim_steps = 100
    iterations = sim_steps - (args.embedding_length - 1) * args.time_delay
    dismat_list = []
    for i in range(iterations):
        steps = [i+args.time_delay*j for j in range(args.embedding_length)]
        dismat_list.append(dismat_from_steps(trajectories, steps))
    return dismat_list

def pers_from_dismat(dismat):
    rips_complex = gudhi.RipsComplex(distance_matrix=dismat,sparse=None)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)
    simplex_tree.compute_persistence()
    pers = simplex_tree.persistence_intervals_in_dimension(0)
    pers = pers[~np.isinf(pers[:, 1])]
    return pers

def compute_wasserstein_signal(dismat_list):
    pers_list = [pers_from_dismat(d) for d in dismat_list]
    wasserstein_signal = [wasserstein_distance(pers_list[0], pers_list[i])
              for i in range(len(pers_list))]
    return np.array(wasserstein_signal)

def matching_distance(matching):
    return sum([abs(bar[0]-bar[1]) for bar in matching])

def compute_matching_signal(dismat_list):
    matching_signal = [matching_distance(perdiver.get_matching_diagram(dismat_list[0], dismat_list[i]))
                                  for i in range(len(dismat_list))]
    return np.array(matching_signal)

###################################
# 4: CREATING THE SIGNALS
###################################

dismat_list = compute_dismat_list(poses, args)
wasserstein_signal = compute_wasserstein_signal(dismat_list)
matching_signal = compute_matching_signal(dismat_list)

###################################
# 5: SAVING THE SIGNALS
###################################

np.save('files/wasserstein.npy', wasserstein_signal)
np.save('files/matching.npy', matching_signal)

###################################
# 5: PLOTTING THE SIGNALS
###################################

fig, ax = plt.subplots(figsize=(7,2))
indices = np.arange(len(dismat_list))
ax.plot(indices, wasserstein_signal, color='c',label='Wasserstein distance')
ax.plot(indices, matching_signal, color='r',label='Matching distance')
ax.set_xlabel('Step')
ax.legend(loc='upper left')
plt.savefig('plots/wass_vs_match_3agents.png')