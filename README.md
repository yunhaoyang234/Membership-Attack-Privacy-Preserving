## Membership Attacks on Reinforcement Learning

### Please note that the code and data is larger than 10 MB.

Code for performing membership attacks and enforcing differential privacy on reinforcement leanring algorithms. The reinforcement leanring algorithms are implemented in [rl-starter-files](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files) and the environments are from [gym-minigrid](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/gym-minigrid). We modified these libraries and included them in our repository:
```
@misc{gym_minigrid,
  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
  title = {Minimalistic Gridworld Environment for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
}

@misc{rl_starter,
  author = {Lucas Willems},
  title = {RL-Starter-Files},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lcswillems/rl-starter-files}},
}

@misc{torch_ac,
  author = {Lucas Willems},
  title = {PyTorch Actor-Critic deep reinforcement learning algorithms: A2C and PPO},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lcswillems/torch-ac}},
}
```

### Installation:
See requirement.txt, please run the code under python 3.7 or later.
Run
`pip install -r requirement.txt`

If you haven't install **gym-minigrid**, please do the following:
```
cd gym-minigrid
pip install -e .
```

If you haven't install **torch-ac**, please do the following:
```
cd torch-ac
pip install -e .
```

If you have problems with runing the code, please try to build a virtual environment and then run the commands above to install required pakages in the virtual environment:
```
pip install virtualenv
virtualenv venv
cd venv
source bin/activate
cd ..
```

### Environments:
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

### Example of Training/Testing RL Models:
#### Train, test, save trajectories on `MiniGrid-FourRooms-v1` environment, using PPO algorithm. Please `cd rl-starter-files` to run the following examples.
1, Train the agent on the `MiniGrid-FourRooms-v1` environment using PPO.
```
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v1 --model FourRoom1 --frames 204800
```

2, Test the agent on the `MiniGrid-FourRooms-v1` and save the trajectories.
```
python -m scripts.train --algo ppo --env MiniGrid-FourRooms-v1 --model FourRoom1 --frames 409600 --test 1
```
The trajectories are saved in `rl-starter-files/storage/FourRoom1/probabilities.csv`

#### Visualization
1, Visualize agent's behavior:
```
python -m scripts.visualize --env MiniGrid-FourRooms-v1 --model FourRoom1
```

2, Visualize the reward plot and other plots:
```
tensorboard --logdir storage/FourRoom1
```
Follow the instructions on terminal and open https://localhost:6006/ on the browser to view the plots.

3, Visualize the multiple plots simultaneously:
Create a new directory `rl-starter-files/storage/fr_plots/` and copy **events.out.tfevents** files to `fr_plots/` directory. An example is [here](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files/storage/fr_plots). Then run:
```
tensorboard --logdir storage/fr_plots
```

**More examples and visualizations can be found on the README in [gym-minigrid](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/gym-minigrid) and [rl-starter-files](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files)**

### Enforcing Differential Privacy
To apply the Gaussian Privacy Module to the trained agent and observe the corresponding utility loss, run
```
python -m scripts.evaluate --env MiniGrid-FourRooms-v1 --model FourRoom1 --sigma 0.1
```
Note that you can change the name of environment and model. Sigma is the noise variance of the Gaussian Privacy Module.

### Code for Reproducing Membership Attack to RL Agent:
Membership attack on RL models under MultiRooms environment: [Reinforcement_Learning.ipynb](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/blob/main/Reinforcement_Learning.ipynb)

## Membership Attacks on Image Classification
#### Dataset
Cifar 10 Dataset that can be imported in Tensorflow or Keras.

#### Code for Reproducing Differential Privacy and Membership Attack:
[Image_Classification.ipynb](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/blob/main/Image_Classification.ipynb)

## Membership Attacks on Machine Translation
#### Dataset
[Multi-30K English-French](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/data).
```
@misc{multi30k,
   title={Multi30K: Multilingual {English-German} Image Descriptions},
   DOI={10.18653/v1/w16-3210},
   journal={Proceedings of the 5th Workshop on Vision and Language},
   publisher={Association for Computational Linguistics},
   author={Elliott, Desmond and Frank, Stella and Sima’an, Khalil and Specia, Lucia},
   year={2016}
}
```

#### Code for Reproducing Differential Privacy and Membership Attack:
[Machine_Translation.ipynb](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/blob/main/Machine_Translation.ipynb)

#### Note:

If you haven't install **Jupyter Notebook**, please go visit [here](https://jupyter.org/install) for installation. If you use `pip`, you can install it with:
```
pip install notebook
```

To run the notebook, go to the directory where your notebook locate at and run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):
```
jupyter-notebook
```
The directory where your jupyter notebook located will be displayed on your browser. Then, please click on the notebook (.ipynb) and run the codes in order.
