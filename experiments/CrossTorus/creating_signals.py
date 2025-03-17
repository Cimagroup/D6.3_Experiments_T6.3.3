###################################
###################################
# EXPERIMENTS: PART 2
# CREATING THE SIGNALS
###################################
###################################

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
from functools import partial
from tslearn import metrics
from tqdm import tqdm
import gudhi
from gudhi.wasserstein import wasserstein_distance
from gudhi import representations
import perdiver.perdiver as perdiver
from sklearn.preprocessing import MinMaxScaler
import pickle
import re
import gc

with open('files/probes_dict.pkl','rb') as file:
    probes_dict = pickle.load(file)

###################################
# 2: ADJUSTING PARAMETERS
###################################

parser = argparse.ArgumentParser(description='Simulation Parameters')
parser.add_argument('--side', type=float, default=10.0, help='Side of the environment')
parser.add_argument('--time_delay', type=int, default=1, help='Time delay to analise simulation intervals')
parser.add_argument('--embedding_length', type=int, default=10, help='Length of the simulation intervals')
parser.add_argument('--epsilon', type=int, default=1, help='Distance between intervals')

args = parser.parse_args([
        '--side', '6.0',
        '--time_delay', '10',
        '--embedding_length', '6',
        '--epsilon', '50'
    ])

###################################
# 3: AUXILIAR FUNCTIONS
###################################

def normangle(angle):
    result = np.mod(angle, 2 * np.pi)
    result[result > np.pi] -= 2 * np.pi
    return result

def custom_distance(vector1, vector2, weights):
    result = 0
    if weights[0] != 0:
        px_diff = np.abs(vector1[0] - vector2[0])
        px_diff = np.minimum(px_diff, args.side - px_diff)
        result += px_diff * weights[0]
    if weights[1] != 0:
        py_diff = np.abs(vector1[1] - vector2[1])
        py_diff = np.minimum(py_diff, args.side - py_diff)
        result += py_diff * weights[1]
    if weights[2] != 0:
        pr_diff = np.abs(vector1[2] - vector2[2])
        pr_diff = np.minimum(pr_diff, 2 * np.pi - pr_diff)
        result += pr_diff * weights[2]
    return result

weights = np.array([2/args.side,2/args.side,1/np.pi])
custom_distance_with_param = partial(custom_distance, weights=weights)

def dismat_from_step(trajectories, step):
    num_agents = trajectories.shape[1]
    dismat = np.zeros((num_agents, num_agents))
    for a in range(num_agents):
        for b in range(a+1):
            tsim = custom_distance_with_param(trajectories[step,a,:],trajectories[step,b,:])
            dismat[a,b] = tsim
    return dismat

def compute_dismat_list(trajectories):
    sim_steps = trajectories.shape[0]
    dismat_list = []
    for step in range(sim_steps):
        dismat_list.append(dismat_from_step(trajectories, step))
    return dismat_list

def dismat_from_steps_1(trajectories, steps):
    num_agents = trajectories.shape[1]
    dismat = np.zeros((num_agents, num_agents))
    for a in range(num_agents):
        for b in range(a+1):
            _, tsim = metrics.dtw_path_from_metric(trajectories[steps,a,:], trajectories[steps,b,:], metric=custom_distance_with_param)
            dismat[a,b] = tsim
    return dismat

def compute_dismat_list_1(trajectories, args):
    sim_steps = trajectories.shape[0]
    iterations = sim_steps - (args.embedding_length - 1) * args.time_delay
    dismat_list = []
    for i in range(iterations):
        steps = [i+args.time_delay*j for j in range(args.embedding_length)]
        dismat_list.append(dismat_from_steps_1(trajectories, steps))
    return dismat_list

def compute_pers_list(dismat_list):
    pers_list = []
    for dismat in dismat_list:
        rips_complex = gudhi.RipsComplex(distance_matrix=dismat)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=0)
        simplex_tree.compute_persistence()
        pers = simplex_tree.persistence_intervals_in_dimension(0)
        pers = pers[np.isfinite(pers[:, 1])]
        pers_list.append(pers)
    return pers_list

def matching_distance(matching):
    return sum([abs(bar[0]-bar[1]) for bar in matching])

def compute_statistics(data):
    median = np.median(data, axis=0)
    percentile_25 = np.percentile(data, 25, axis=0)
    percentile_75 = np.percentile(data, 75, axis=0)
    return median, percentile_25, percentile_75

def plot_signals_summary(types, signals, title, output_path):
    subset_1 = signals[types == 0]
    subset_2 = signals[types == 1]
    subset_3 = signals[types == 2]

    median_1, p25_1, p75_1 = compute_statistics(subset_1)
    median_2, p25_2, p75_2 = compute_statistics(subset_2)
    median_3, p25_3, p75_3 = compute_statistics(subset_3)

    plt.figure(figsize=(12, 6))

    plt.fill_between(range(subset_1.shape[2]), p25_1.squeeze(), p75_1.squeeze(), color='blue', alpha=0.3)
    plt.plot(median_1.squeeze(), color='blue', label='HL')

    plt.fill_between(range(subset_2.shape[2]), p25_2.squeeze(), p75_2.squeeze(), color='red', alpha=0.3)
    plt.plot(median_2.squeeze(), color='red', label='ORCA')

    plt.fill_between(range(subset_3.shape[2]), p25_3.squeeze(), p75_3.squeeze(), color='green', alpha=0.3)
    plt.plot(median_3.squeeze(), color='green', label='SocialForce')

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(output_path, dpi='figure', bbox_inches='tight')
    plt.close()
    
def clean_text(text):
    text1 = re.sub(r"[()]", "", text)
    result = text1.replace(" ", "_")
    return result

def combine_signals(dictionary,key1,key2):
    signal1 = dictionary[key1]
    signal2 = dictionary[key2]
    len1 = signal1.shape[2]
    len2 = signal2.shape[2]
    lenf = min(len1,len2)
    dictionary[f"{key1} + {key2}"] = np.concatenate([signal1[:,:,:lenf],signal2[:,:,:lenf]],axis=1)

###################################
# 4: CREATING THE SIGNALS
###################################

signals_dict = {}

signals_eff = np.mean(probes_dict['Efficacies'], axis=2)
scaler_eff = MinMaxScaler()
scaled_eff = scaler_eff.fit_transform(signals_eff.T).T
signals_dict['Efficacy'] = scaled_eff[:, np.newaxis, :]
signals_saf = np.mean(probes_dict['Safeties'], axis=2)
scaler_saf = MinMaxScaler()
scaled_saf = scaler_saf.fit_transform(signals_saf.T).T
signals_dict['Safety'] = scaled_saf[:, np.newaxis, :]
with open('files/signals_dict.pkl','wb') as file:
    pickle.dump(signals_dict,file)

num_simulations = probes_dict['Poses'].shape[0]
signals_ent = None
signals_was = None
signals_mat = None
signals_ent_dtw = None
signals_was_dtw = None
signals_mat_dtw = None

for i in tqdm(range(num_simulations), desc="Progress: ", leave = False):
    trajectories = probes_dict['Poses'][i].copy()
    trajectories[:,:,2] = normangle(trajectories[:,:,2])
    #Just for one step
    dismat_list = compute_dismat_list(trajectories)
    pers_list = compute_pers_list(dismat_list)
    pe = representations.Entropy()
    signal_ent = pe.fit_transform(pers_list)
    scaler_ent = MinMaxScaler()
    scaled_ent = scaler_ent.fit_transform(signal_ent).reshape((1,1,-1))
    if signals_ent is None:
        signals_ent = scaled_ent.copy()
    else:
        signals_ent = np.concatenate((signals_ent, scaled_ent), axis=0)
    signal_was = np.array([wasserstein_distance(pers_list[i],pers_list[i+args.epsilon]) 
        for i in range(len(pers_list)-args.epsilon)]).reshape((-1,1))
    scaler_was = MinMaxScaler()
    scaled_was = scaler_was.fit_transform(signal_was).reshape((1,1,-1))
    if signals_was is None:
        signals_was = scaled_was.copy()
    else:
        signals_was = np.concatenate((signals_was, scaled_was), axis=0)
    signal_mat = np.array([matching_distance(perdiver.get_matching_diagram(dismat_list[i],dismat_list[i+args.epsilon])) 
        for i in range(len(dismat_list)-args.epsilon)]).reshape((-1,1))
    scaler_mat = MinMaxScaler()
    scaled_mat = scaler_mat.fit_transform(signal_mat).reshape((1,1,-1))
    if signals_mat is None:
        signals_mat = scaled_mat.copy()
    else:
        signals_mat = np.concatenate((signals_mat, scaled_mat), axis=0)
    signals_dict['Entropy'] = signals_ent
    signals_dict['Wasserstein'] = signals_was
    signals_dict['Matching'] = signals_mat
    #For many steps
    dismat_list_1 = compute_dismat_list_1(trajectories, args)
    pers_list_1 = compute_pers_list(dismat_list_1)
    pe_1 = representations.Entropy()
    signal_ent_1 = pe_1.fit_transform(pers_list_1)
    scaler_ent_1 = MinMaxScaler()
    scaled_ent_1 = scaler_ent_1.fit_transform(signal_ent_1).reshape((1,1,-1))
    if signals_ent_dtw is None:
        signals_ent_dtw = scaled_ent_1.copy()
    else:
        signals_ent_dtw = np.concatenate((signals_ent_dtw, scaled_ent_1), axis=0)
    signal_was_1 = np.array([wasserstein_distance(pers_list_1[i],pers_list_1[i+args.epsilon]) 
        for i in range(len(pers_list_1)-args.epsilon)]).reshape((-1,1))
    scaler_was_1 = MinMaxScaler()
    scaled_was_1 = scaler_was_1.fit_transform(signal_was_1).reshape((1,1,-1))
    if signals_was_dtw is None:
        signals_was_dtw = scaled_was_1.copy()
    else:
        signals_was_dtw = np.concatenate((signals_was_dtw, scaled_was_1), axis=0)
    signal_mat_1 = np.array([matching_distance(perdiver.get_matching_diagram(dismat_list_1[i], dismat_list_1[i+args.epsilon]))
        for i in range(len(dismat_list_1)-args.epsilon)]).reshape((-1,1))
    scaler_mat_1 = MinMaxScaler()
    scaled_mat_1 = scaler_mat_1.fit_transform(signal_mat_1).reshape((1,1,-1))
    if signals_mat_dtw is None:
        signals_mat_dtw = scaled_mat_1.copy()
    else:
        signals_mat_dtw = np.concatenate((signals_mat_dtw, scaled_mat_1), axis=0)
    signals_dict['Entropy (DTW)'] = signals_ent_dtw
    signals_dict['Wasserstein (DTW)'] = signals_was_dtw
    signals_dict['Matching (DTW)'] = signals_mat_dtw
    with open('files/signals_dict.pkl','wb') as file:
        pickle.dump(signals_dict,file)
    del trajectories
    del dismat_list, pers_list, pe
    del signal_ent, signal_was, signal_mat
    del scaler_ent, scaler_was, scaler_mat
    del scaled_ent, scaled_was, scaled_mat
    del dismat_list_1, pers_list_1, pe_1
    del signal_ent_1, signal_was_1, signal_mat_1
    del scaler_ent_1, scaler_was_1, scaler_mat_1
    del scaled_ent_1, scaled_was_1, scaled_mat_1
    gc.collect()

###################################
# 5: PLOTTING THE SIGNALS
###################################

for key in signals_dict.keys():
    plot_signals_summary(probes_dict['Types'], signals_dict[key], 
                         f'Summary of {key} Signals', 
                         f'plots/signals_{clean_text(key)}.png')

###################################
# 6: COMBINED SIGNALS
###################################

combine_signals(signals_dict, 'Efficacy','Safety')
key_list = ['Entropy','Wasserstein','Matching',
    'Entropy (DTW)','Wasserstein (DTW)','Matching (DTW)']
for key in key_list:
    combine_signals(signals_dict, 'Efficacy + Safety', key)
with open('files/signals_dict.pkl','wb') as file:
    pickle.dump(signals_dict,file)
