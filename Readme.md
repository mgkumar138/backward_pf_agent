# Predictive Coding of Reward in the Hippocampus - Simulation Code

This repository contains the code to replicate the simulations presented in the paper:

> **Predictive Coding of Reward in the Hippocampus**  
> Mohammad Yaghoubi, M Ganesh Kumar, Andres Nieto-Posadas, Coralie-Anne Mosser, Thomas Gisiger, Émmanuel Wilson, Cengiz Pehlevan, Sylvain Williams, Mark P. Brandon.  
> Nature 2025

## Overview

The main idea is to optimize hippocampal place cell centers using the Temporal Difference (TD) error, and study the resulting shifting dynamics. Place cell centers are tracked as they shift in response to learning about rewards.

## Code Structure

There are two main Jupyter notebooks in this repository:

1. **`backward_shift_fields.ipynb`**  
   - *Value estimation learning only*:  
     Place cells feed into a critic and the agent follows a direct path from the start state to the reward state.

2. **`backward_shift_fields_policy.ipynb`**  
   - *Policy learning*:  
     The agent must learn to navigate from the start to the reward location by choosing among three actions: left (`action=0`), right (`action=1`), or stay (`action=2`).

Hyperparameter runs with different numbers of place cells, place cell widths, and reward discount factors (gamma) are provided in the `large_runs/` folder.

## Main Result

We find that place cell centers systematically shift backwards from the reward to the start state during learning, replicating key experimental findings.

## Citation

If you find this repository useful for your research, please cite:

```
@article{yaghoubi2025predictive,
  author = {Yaghoubi, Mohammad and Kumar, M Ganesh and Nieto-Posadas, Andres and Mosser, Coralie-Anne and Gisiger, Thomas and Wilson, Émmanuel and Pehlevan, Cengiz and Williams, Sylvain and Brandon, Mark P.},
  title = {Predictive Coding of Reward in the Hippocampus},
  journal = {Nature},
  year = {2025},
  note = {In press}
}
```