# Membership Attacks on Reinforcement Learning

Code for performing membership attacks on reinforcement leanring algorithms. The reinforcement leanring algorithms are implemented in [rl-starter-files](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files) and the environments are from [gym-minigrid](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/gym-minigrid).

## Installation:
See requirement.txt, please run the code under python 3.7 or later.
Run
`pip install -r requirement.txt`

If you haven't install **gym-minigrid**, please do the following:
```
cd gym-minigrid
pip install -e .
```

If you haven't install **rl-starter-files**, please do the following:
```
cd rl-starter-files
pip install -r requirements.txt
```

If you haven't install **torch-ac**, please do the following:
```
cd torch-ac
pip install -e .
```

## Environments:
- `MiniGrid-FourRooms-v1` - Four room environment where the agent always starts from room 1.
- `MiniGrid-FourRooms-v2` - Four room environment where the agent always starts from room 2.
- `MiniGrid-FourRooms-v3` - Four room environment where the agent always starts from room 3.

- `MiniGrid-MultiRoom-N2-v00`- Multi-room environment with seed range [0, 4], each seed correspondes to a unique setting (eg. room size, room position, agent starting position, target position, etc.)
- `MiniGrid-MultiRoom-N2-v0`- Multi-room environment with seed 0.
- `MiniGrid-MultiRoom-N2-v1`- Multi-room environment with seed 1.
- `MiniGrid-MultiRoom-N2-v2`- Multi-room environment with seed 2.
- `MiniGrid-MultiRoom-N2-v5`- Multi-room environment with seed 5.
- `MiniGrid-MultiRoom-N2-v6`- Multi-room environment with seed 6.
- `MiniGrid-MultiRoom-N2-v7`- Multi-room environment with seed 7.

## Example of Training/Testing RL Models:
#### Train, test, save trajectories on `MiniGrid-FourRooms-v1` environment, using PPO algorithm.
1, Train the agent on the `MiniGrid-FourRooms-v1` environment using PPO.
```
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v1 --model FourRoom1 --frames 204800
```

2, Test the agent on the `MiniGrid-FourRooms-v1` and save the trajectories.
```
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v1 --model FourRoom1 --frames 409600 --test 1
```
The trajectories are saved in `rl-starter-files/storage/FourRoom1/probabilities.csv`

#### Test and save trajectories on `MiniGrid-FourRooms-v1` environment, using PPO algorithm with Dirichlet Protection Mechanism.
Test the agent on the `MiniGrid-FourRooms-v1` and save the trajectories. Apply Dirichlet Protection with **k = 1**.
```
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v1 --model FourRoom_protected --frames 204800
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v1 --model FourRoom_protected --frames 409600 --test 1 --k 1
```
The trajectories are saved in `rl-starter-files/storage/FourRoom_protected/probabilities.csv`

#### Visualization
1, Visualize agent's behavior:
```
python -m scripts.visualize --env MiniGrid-FourRooms-v1 --model FourRoom1
```

2, Visualize the reward plot and other plots:
```
tensorboard --logdir rl-starter-files/storage/FourRoom1
```

3, Visualize the multiple plots simultaneously:
Create a new directory `rl-starter-files/storage/fr_plots/` and copy **events.out.tfevents** files to `fr_plots/` directory. An example is [here](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files/storage/mr_plots). Then run:
```
tensorboard --logdir rl-starter-files/storage/fr_plots
```

**More examples and visualizations can be found on the README in [gym-minigrid](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/gym-minigrid) and [rl-starter-files](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files)**

## Examples of Membership Attack:
1, Membership attack on supervised learning: [Mem_att_toy_cifar10.ipynb](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/blob/main/Mem_att_toy_cifar10.ipynb).

2, Membership attack on RL models under FourRooms environment: [RL_Men_att_four_room.ipynb](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/blob/main/RL_Men_att_four_room.ipynb)

3, Membership attack on RL models under MultiRooms environment: [RL_Men_att_multi_rooms.ipynb](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/blob/main/RL_Men_att_multi_rooms.ipynb)

#### Note:

If you haven't install **Jupyter Notebook**, please go visit [here](https://jupyter.org/install) for installation. If you use `pip`, you can install it with:
```
pip install notebook
```

To run the notebook, go to the directory where your notebook locate at and run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):
```
jupyter notebook
```

