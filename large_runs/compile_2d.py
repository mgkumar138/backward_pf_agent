#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Hyperparameters
gammas = [0.0, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
Ns = [5, 10, 25, 50, 100, 250, 500, 1000]
sigma_inits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Fixed typo (duplicate 0.1 in your original)
w_init = 0.0
eta0 = 0.1
nstates = 10
seed = 0

# Initialize data storage
data_points = np.zeros((len(gammas), len(sigma_inits), 3))  # Preallocate for speed

# Load data for all combinations
for g, gamma in enumerate(gammas):
    for s, sigma_init in enumerate(sigma_inits):
        try:
            data = np.load(
                f'./data/dcells_{sigma_init}sigma_{w_init}w_{eta0}eta0_{N}N_{gamma}gamma_{nstates}states_{seed}seed_td.npz',
                allow_pickle=True
            )
            reward_cells = data['df_perc_reward_cells']
            approach_cells = data['df_perc_reward_approach_cells']
            screen_cells = data['df_perc_screen_cells']
            data_points[g,s] = reward_cells, approach_cells, screen_cells
        except FileNotFoundError:
            print(f"File not found for gamma={gamma}, sigma_init={sigma_init}. Skipping...")


#%%
# Create figure with 3 subplots
fig = plt.figure(figsize=(4, 9))
titles = ['Reward Cells', 'Approach Cells', 'Screen Cells']
metrics = [2, 3, 4]  # Columns in data_points for reward, approach, screen

for i, (title, metric_col) in enumerate(zip(titles, metrics), 1):
    ax = fig.add_subplot(3, 1, i, projection='3d')
    scatter = ax.scatter(
        data_points[:, 0],  # gamma
        data_points[:, 1],  # N
        data_points[:, 2],  # sigma_init
        c=data_points[:, metric_col],  # metric (color)
        cmap='viridis',
        s=50
    )
    ax.set_xlabel('Gamma')
    ax.set_ylabel('N')
    ax.set_yscale('log')
    ax.set_zlabel('Sigma Init')
    ax.set_zscale('log')
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label='Percentage of Cells')

plt.tight_layout()
plt.show()