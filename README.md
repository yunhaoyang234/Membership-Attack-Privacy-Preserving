# Membership Inference Attacks on Actor-Critic Algorithm

Code for performing membership inference attacks on actor-critic algorithms. The reinforcement leanring algorithms are implemented in [rl-starter-files](https://github.com/lcswillems/rl-starter-files) and the environments are from [gym-minigrid](https://github.com/maximecb/gym-minigrid).


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
- `MiniGrid-MultiRoom-N2-v00`- Multi-room environment with seed range [0, 10], each seed correspondes to a unique setting (eg. room size, room position, agent starting position, target position, etc.)
- `MiniGrid-MultiRoom-N2-v0`- Multi-room environment with seed 0.
- `MiniGrid-MultiRoom-N2-v1`- Multi-room environment with seed 1.
- `MiniGrid-MultiRoom-N2-v2`- Multi-room environment with seed 2.
- `MiniGrid-MultiRoom-N2-v5`- Multi-room environment with seed 5.
- `MiniGrid-MultiRoom-N2-v6`- Multi-room environment with seed 6.
- `MiniGrid-MultiRoom-N2-v7`- Multi-room environment with seed 7.
- We have created the environment to seed 19.

## Example of Training/Testing RL Models:
#### Train, test, save trajectories on `MiniGrid-MultiRoom-N2-v00` environment, using PPO algorithm.
1, Train the agent on the `MiniGrid-MultiRoom-N2-v00` environment using PPO.
```
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N2-v00 --model mr --frames 204800
```

2, Test the agent on the `MiniGrid-MultiRoom-N2-v00` and save the trajectories.
```
python -m scripts.evaluate --env MiniGrid-MultiRoom-N2-v00 --model mr
```
The trajectories are saved in `rl-starter-files/storage/mr/value_traj.csv`

#### Train, Test and save trajectories on `MiniGrid-MultiRoom-N2-v00` environment, using PPO algorithm with DP-SGD.
Test the agent on the `MiniGrid-FourRooms-v1` and save the trajectories. Apply Dirichlet Protection with **k = 1**.
```
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N2-v00 --model mr_dp --frames 204800 --sigma 0.8
python -m scripts.evaluate --env MiniGrid-MultiRoom-N2-v00 --model mr_dp
```
The trajectories are saved in `rl-starter-files/storage/mr/value_traj.csv`

#### Train with different advantage estimation methods
1, TD-Error
```
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N2-v00 --model mr --frames 204800 --gae-lambda 0
```

2, N-Step Advantage
```
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N2-v00 --model mr --frames 204800 --gae-lambda 3 (or any integer)
```

3, Generalized Advantage Estimation
```
python -m scripts.train --algo ppo --env MiniGrid-MultiRoom-N2-v00 --model mr --frames 204800 --gae-lambda 0.95 (or any float between 0 and 1)
```

#### Visualization
1, Visualize agent's behavior:
```
python -m scripts.visualize --env MiniGrid-MultiRoom-N2-v00 --model mr
```

2, Visualize the reward plot and other plots:
```
tensorboard --logdir rl-starter-files/storage/mr
```

**More examples and visualizations can be found on the README in [gym-minigrid] and [rl-starter-files]**

## Examples of Membership Attack:
1, Membership attack on supervised learning: [Mem_att_toy_cifar10.ipynb].

2, Membership attack on RL models under FourRooms environment: [RL_Men_att_four_room.ipynb]

3, Membership attack on RL models under MultiRooms environment: [RL_Men_att_multi_rooms.ipynb]

#### Note:

If you haven't install **Jupyter Notebook**, please go visit [here](https://jupyter.org/install) for installation. If you use `pip`, you can install it with:
```
pip install notebook
```

To run the notebook, go to the directory where your notebook locate at and run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):
```
jupyter notebook
```

