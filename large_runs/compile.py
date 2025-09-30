#%%
import numpy as np
import matplotlib.pyplot as plt


gammas = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3,0.4, 0.5, 0.6,0.7,0.8, 0.9, 0.95,0.99, 1.0] # 0.0 0.001 0.01 0.1 0.25 0.5 0.75 0.9 0.95 1
N = 1000
sigma_init = 0.5
w_init = 0.0
eta0 = 0.1
nstates = 10
seed = 0

df = np.zeros((len(gammas), 4))

for i, gamma in enumerate(gammas):
    try:                 #data/dcells_0.1sigma_0.0w_0.1eta0_5N_0.0gamma_5states.npz
        data = np.load(f'./data/dcells_{sigma_init}sigma_{w_init}w_{eta0}eta0_{N}N_{gamma}gamma_{nstates}states_{seed}seed_td.npz', allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found for gamma {gamma}. Skipping...")

    df_perc_reward_cells = data['df_perc_reward_cells']
    df_perc_reward_approach_cells = data['df_perc_reward_approach_cells']
    df_perc_screen_cells = data['df_perc_screen_cells']
    tdf = data['tdf']

    df[i] = np.array([df_perc_reward_cells, df_perc_reward_approach_cells, df_perc_screen_cells, tdf])

plt.figure(figsize=(3,2))
ax1 = plt.gca()
l1, = ax1.plot(gammas, df[:, 0], label='Reward')
l2, = ax1.plot(gammas, df[:, 1], label='Approach')
l3, = ax1.plot(gammas, df[:, 2], label='Screen')
ax2 = ax1.twinx()
l4, = ax2.plot(gammas, df[:, 3], label='$L_v$', color='red')

# Combine legends from both axes
lines = [l1, l2, l3, l4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=8)

ax1.set_xlabel('$\gamma$')
ax1.set_ylabel('$\Delta$ cells (%)')
ax2.set_ylabel('Final Value Loss')
ax2.set_yscale('log')
plt.title('Reward Discount Factor')
ax1.grid(True,alpha=0.3)
plt.savefig(f'./figs/alt_soln/gamma_{N}N_{sigma_init}sigma.svg', bbox_inches='tight')

print('Best gamma:', gammas[np.argmin(df[:, 3])])


#%%

gamma = 0.95
Ns = [10, 25, 50, 100, 250, 500, 1000] # 1,2, 5 10 25 50 100 250 500 1000
sigma_init = 0.5

df = np.zeros((len(Ns), 4))

for i, N in enumerate(Ns):
    try:                 #data/dcells_0.1sigma_0.0w_0.1eta0_5N_0.0gamma_5states.npz
        data = np.load(f'./data/dcells_{sigma_init}sigma_{w_init}w_{eta0}eta0_{N}N_{gamma}gamma_{nstates}states_{seed}seed_td.npz', allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found for N {N}. Skipping...")

    df_perc_reward_cells = data['df_perc_reward_cells']
    df_perc_reward_approach_cells = data['df_perc_reward_approach_cells']
    df_perc_screen_cells = data['df_perc_screen_cells']
    tdf = data['tdf']

    df[i] = np.array([df_perc_reward_cells, df_perc_reward_approach_cells, df_perc_screen_cells, tdf])

plt.figure(figsize=(3,2))
ax1 = plt.gca()
l1, = ax1.plot(Ns, df[:, 0], label='Reward')
l2, = ax1.plot(Ns, df[:, 1], label='Approach')
l3, = ax1.plot(Ns, df[:, 2], label='Screen')
ax2 = ax1.twinx()
l5, = ax2.plot(Ns, df[:, 3], label='$L_v$', color='red')

# Combine legends from both axes
lines = [l1, l2, l3, l5]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=8)

ax1.set_xscale('log')
ax1.set_xlabel('$N$')
ax1.set_ylabel('$\Delta$ cells (%)')
ax2.set_ylabel('Final Value Loss')
ax2.set_yscale('log')
plt.title('Number of Place Cells')
ax1.grid(True,alpha=0.3)
plt.savefig(f'./figs/alt_soln/N_{gamma}g_{sigma_init}sigma.svg', bbox_inches='tight')

print('Best N:', Ns[np.argmin(df[:, 3])])

# %%
gamma = 0.95
N = 1000
sigma_inits = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


df = np.zeros((len(sigma_inits),4))

for i, sigma_init in enumerate(sigma_inits):
    try:                 #data/dcells_0.1sigma_0.0w_0.1eta0_5N_0.0gamma_5states.npz
        data = np.load(f'./data/dcells_{sigma_init}sigma_{w_init}w_{eta0}eta0_{N}N_{gamma}gamma_{nstates}states_{seed}seed_td.npz', allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found for sigma {sigma_init}. Skipping...")

    df_perc_reward_cells = data['df_perc_reward_cells']
    df_perc_reward_approach_cells = data['df_perc_reward_approach_cells']
    df_perc_screen_cells = data['df_perc_screen_cells']
    tdf = data['tdf']

    df[i] = np.array([df_perc_reward_cells, df_perc_reward_approach_cells, df_perc_screen_cells, tdf])

plt.figure(figsize=(3,2))
ax1 = plt.gca()
l1, = ax1.plot(sigma_inits, df[:, 0], label='Reward')
l2, = ax1.plot(sigma_inits, df[:, 1], label='Approach')
l3, = ax1.plot(sigma_inits, df[:, 2], label='Screen')
ax2 = ax1.twinx()
l5, = ax2.plot(sigma_inits, df[:, 3], label='$L_v$', color='red')

# Combine legends from both axes
lines = [l1, l2, l3, l5]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4, fontsize=8)

ax1.set_xscale('log')
ax1.set_xlabel('$\sigma$')
ax1.set_ylabel('$\Delta$ cells (%)')
ax2.set_ylabel('Final Value Loss')
ax2.set_yscale('log')
plt.title('Place Cell Width')
ax1.grid(True,alpha=0.3)
# plt.savefig(f'./figs/alt_soln/sigma_{N}N_{gamma}g.png', bbox_inches='tight', dpi=300)
plt.savefig(f'./figs/alt_soln/sigma_{N}N_{gamma}g.svg', bbox_inches='tight')

print('Best sigma:', sigma_inits[np.argmin(df[:, 3])])