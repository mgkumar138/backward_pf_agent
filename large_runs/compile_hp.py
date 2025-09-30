#%%
import os
import numpy as np
from glob import glob

import matplotlib.pyplot as plt

# Hyperparameter values (should match your experiment)
# gammas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
# Ns = [5, 10, 50, 100, 500, 1000]
# sigma_inits = [0.1, 0.25, 0.5, 1.0]


w_init = 0.0
eta0 = 0.1
seed = 0
nstates = 5

data_dir = './data'

def load_results():
    results = []
    files = glob(os.path.join(data_dir, 'dcells_*states.npz'))
    for f in files:
        # Parse hyperparameters from filename
        fname = os.path.basename(f)
        try:
            sigma_init = float(fname.split('_')[1].replace('sigma', ''))
            w = float(fname.split('_')[2].replace('w', ''))
            eta = float(fname.split('_')[3].replace('eta0', ''))
            N = int(fname.split('_')[4].replace('N', ''))
            gamma = float(fname.split('_')[5].replace('gamma', ''))
            nstates_ = int(fname.split('_')[6].replace('states.npz', ''))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
        arr = np.load(f)
        results.append({
            'gamma': gamma,
            'N': N,
            'sigma_init': sigma_init,
            'df_perc_reward_cells': arr['df_perc_reward_cells'],
            'df_perc_reward_approach_cells': arr['df_perc_reward_approach_cells'],
            'df_perc_screen_cells': arr['df_perc_screen_cells'],
        })
    return results

def get_delta(values, param_list):
    # Sort by param_list, then compute delta (difference between consecutive)
    sorted_idx = np.argsort(param_list)
    sorted_vals = np.array(values)[sorted_idx]
    delta = np.diff(sorted_vals)
    return delta, np.array(param_list)[sorted_idx][1:]

def plot_delta_combined(results, param_name, param_values):
    # Gather values for each param for all cell types
    reward_vals, approach_vals, screen_vals, params = [], [], [], []
    for v in param_values:
        for r in results:
            if np.isclose(r[param_name], v):
                reward_vals.append(r['df_perc_reward_cells'])
                approach_vals.append(r['df_perc_reward_approach_cells'])
                screen_vals.append(r['df_perc_screen_cells'])
                params.append(v)
                break
    # Compute deltas
    delta_reward, x = get_delta(reward_vals, params)
    delta_approach, _ = get_delta(approach_vals, params)
    delta_screen, _ = get_delta(screen_vals, params)
    # Plot
    plt.figure()
    plt.plot(x, delta_reward, marker='o', color='blue', label='Reward Cells')
    plt.plot(x, delta_approach, marker='o', color='orange', label='Reward Approach Cells')
    plt.plot(x, delta_screen, marker='o', color='green', label='Screen Cells')
    plt.xlabel(param_name)
    plt.ylabel('Delta % Cells')
    plt.title(f'Delta % Cells vs {param_name}')
    plt.legend()
    plt.grid(True)

# def main():
#     results = load_results()
#     param_dict = {
#         'gamma': gammas,
#         'sigma_init': sigma_inits,
#         'N': Ns
#     }
#     for param_name, param_values in param_dict.items():
#         plot_delta_combined(results, param_name, param_values)
#     plt.show()



# Define hyperparameter groups
hyperparam_groups = [
    {
        'name': 'Varying gamma',
        'param_name': 'gamma',
        'param_values': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1],
        'fixed': {'N': 1000, 'sigma_init': 0.5}
    },
    {
        'name': 'Varying N',
        'param_name': 'N',
        'param_values': [5, 10, 50, 100, 500, 1000],
        'fixed': {'gamma': 0.9, 'sigma_init': 0.5}
    },
    {
        'name': 'Varying sigma_init',
        'param_name': 'sigma_init',
        'param_values': [0.1, 0.25, 0.5, 1.0],
        'fixed': {'gamma': 0.9, 'N': 1000}
    }
]

def filter_results(results, fixed_params):
    filtered = []
    for r in results:
        match = True
        for k, v in fixed_params.items():
            if not np.isclose(r[k], v):
                match = False
                break
        if match:
            filtered.append(r)
    return filtered

def main():
    results = load_results()
    for group in hyperparam_groups:
        filtered = filter_results(results, group['fixed'])
        if not filtered:
            print(f"No results found for group: {group['name']}")
            continue
        plot_delta_combined(filtered, group['param_name'], group['param_values'])
        plt.title(f"Delta % Cells vs {group['param_name']} ({group['name']})")
    plt.show()

main()
