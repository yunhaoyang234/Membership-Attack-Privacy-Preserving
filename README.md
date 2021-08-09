# Membership Attacks on Reinforcement Learning

Code for performing membership attacks on reinforcement leanring algorithms. The reinforcement leanring algorithms are implemented in [rl-starter-files](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/rl-starter-files) and the environments are from [gym-minigrid](https://github.com/yunhaoyang234/Membership-Attack-Privacy-Preserving/tree/main/gym-minigrid).

## Set up:
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

## Experiments:
#### Geo-query Question-Answering Experiment
```bash
$ python main.py \
	 --experiment 'geo'\
         --train_path 'data/geo_train.tsv'\
         --dev_path 'data/geo_dev.tsv'\
         --epochs 10\
         --lr 0.001\
         --hidden_size 200\
    	 --num_filters 2\
    	 --dropout 0.2\
    	 --embedding_dim 150
```
The token-level accuracy and denotation accuracy for the development set will be printed out. Set '--plot 1' to display the latent space clustering result. The results may vary each time, you can run multiple times and get an averaged result.

#### Bilingual Sentence Pairs Translation Experiment
```bash
$ python main.py \
	 --experiment 'translate'\
         --epochs 10\
         --lr 0.001\
         --hidden_size 250\
    	 --num_filters 2\
    	 --dropout 0.1\
         --plot 1\
    	 --embedding_dim 200
```
The BLEU score for the test set will be printed out. Set '--plot 1' to display the latent space clustering result.
