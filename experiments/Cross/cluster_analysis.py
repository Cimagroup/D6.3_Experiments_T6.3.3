###################################
###################################
# EXPERIMENTS: PART 3
# CLUSTER ANALYSIS
###################################
###################################

###################################
# 1: IMPORTING MODULES
###################################

import numpy as np
from sklearn.metrics import silhouette_samples
from umap import UMAP
import matplotlib.pyplot as plt
import pickle
from tslearn import metrics
import gc
import re
import time

with open('files/probes_dict.pkl','rb') as file:
    types = pickle.load(file)['Types']

with open('files/signals_dict.pkl','rb') as file:
    signals_dict = pickle.load(file)

###################################
# 2: AUXILIAR FUNCTIONS
###################################

def plot_umap(types,projection,title,output_path):
    color_map = {0: "blue", 1: "red", 2: "green"}
    colors = [color_map[t] for t in types] 
    plt.figure(figsize=(8, 6))
    plt.scatter(projection[:, 0], projection[:, 1], c=colors, s=50, alpha=0.7, edgecolors="k")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="HL"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="ORCA"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="SocialForce"),
    ])
    plt.savefig(output_path, dpi='figure', bbox_inches='tight')
    plt.close()

def clean_text(text):
    text1 = re.sub(r"[()]", "", text)
    result = text1.replace(" ", "_")
    return result

###################################
# 3: SILHOUETTE SCORES AND UMAP
###################################

silhouettes_dict = {}
umap_dict = {}
num_keys = len(signals_dict)
for key, value in signals_dict.items():
    print(f'Computing silhouette scores for {key} signals')
    num_signals, _, num_steps = value.shape
    dismat = np.zeros((num_signals, num_signals))
    sil = np.zeros((num_signals,num_steps))
    start_time = time.time()
    for step in range(num_steps):
        for i in range(num_signals):
            for j in range(i + 1, num_signals):
                d = np.sum(abs(value[i, :, step] - value[j, :, step]))
                dismat[i, j] += d
                dismat[j, i] += d
        sil[:,step] = silhouette_samples(dismat, types, metric='precomputed')
        if (step+1) % 100 == 0:
            print(f'Step {step+1}/{num_steps}')
            end_time = time.time()
            avg_time_per_iteration = (end_time - start_time) / (step+1)
            estimated_time_left = avg_time_per_iteration * (num_steps-step-1)
            print(f"Average time for iteration: {avg_time_per_iteration:.2f} seconds")
            print(f"Estimated time remaining: {estimated_time_left / 60:.2f} minutes")
    silhouettes_dict[key] = sil
    umap_model = UMAP(n_components=2, metric="precomputed", n_epochs=1000, random_state=2025)
    embedding = umap_model.fit_transform(dismat)
    umap_dict[key] = embedding
    with open('files/silhouettes_dict.pkl','wb') as file:
        pickle.dump(silhouettes_dict,file)
    with open('files/umap_dict.pkl','wb') as file:
        pickle.dump(umap_dict,file)
    del num_signals, num_steps
    del dismat, sil
    del umap_model, embedding
    gc.collect()

###################################
# 5: PLOTTING
###################################

unisignals = ['Efficacy','Safety','Entropy','Wasserstein','Matching','Entropy (DTW)','Wasserstein (DTW)','Matching (DTW)']
comsignals = ['Efficacy + Safety', 'Efficacy + Safety + Entropy', 'Efficacy + Safety + Wasserstein', 'Efficacy + Safety + Matching', 
              'Efficacy + Safety + Entropy (DTW)', 'Efficacy + Safety + Wasserstein (DTW)', 'Efficacy + Safety + Matching (DTW)']

plt.figure(figsize=(8, 6))
for key in unisignals:
    matrix = silhouettes_dict[key]
    mean_signal = np.mean(matrix, axis=0)
    plt.plot(mean_signal, label=key)
plt.xlabel('Step')
plt.ylabel('Silhouette score')
plt.title('Evolution of silhouette score')
plt.grid(True)
plt.legend()
plt.savefig('plots/silhouette_scores_uni.png', dpi='figure', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
for key in comsignals:
    matrix = silhouettes_dict[key]
    mean_signal = np.mean(matrix, axis=0)
    plt.plot(mean_signal, label=key)
plt.xlabel('Step')
plt.ylabel('Silhouette score')
plt.title('Evolution of silhouette score')
plt.grid(True)
plt.legend()
plt.savefig('plots/silhouette_scores_com.png', dpi='figure', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
for key in unisignals:
    matrix = silhouettes_dict[key]
    mean_signal = np.mean(matrix>0, axis=0)
    plt.plot(mean_signal, label=key)
plt.xlabel('Step')
plt.ylabel('Silhouette score')
plt.title('Fraction of signals with positive silhouette')
plt.grid(True)
plt.legend()
plt.savefig('plots/silhouette_positives_uni.png', dpi='figure', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
for key in comsignals:
    matrix = silhouettes_dict[key]
    mean_signal = np.mean(matrix>0, axis=0)
    plt.plot(mean_signal, label=key)
plt.xlabel('Step')
plt.ylabel('Silhouette score')
plt.title('Fraction of signals with positive silhouette')
plt.grid(True)
plt.legend()
plt.savefig('plots/silhouette_positives_com.png', dpi='figure', bbox_inches='tight')
plt.close()

for key, value in umap_dict.items():
    plot_umap(types,value,f"UMAP projection of {key} signals",f"plots/umap_{clean_text(key)}.png")
