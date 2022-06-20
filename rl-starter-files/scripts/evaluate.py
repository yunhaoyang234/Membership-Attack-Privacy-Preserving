import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import numpy as np
import pandas as pd

import utils


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=20,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=1,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--sigma", type=float, default=0, help="sigma for Gaussian mechanism")
parser.add_argument("--label", type=int, default=1, help="1 for in, 0 for out")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    env = utils.make_env(args.env, args.seed + 10000 * i)
    envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, num_envs=args.procs,
                    use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

traj_dir = 'storage/' + args.model + '/critic_traj.csv'
try:
    df = pd.read_csv(traj_dir)
except FileNotFoundError: 
    df = pd.DataFrame()
    df['value'] = []
    df['reward'] = []
    df['env'] = []
    df['label'] = []
    df['corr'] = []

value_list, reward_list, env_list, label_list, corr_list = list(df['value']), list(df['reward']), list(df['env']), list(df['label']), list(df['corr'])

# Add noise
if args.sigma > 0:
    for param in agent.acmodel.actor.state_dict():
        size = agent.acmodel.actor.state_dict()[param].shape
        agent.acmodel.actor.state_dict()[param] += torch.Tensor(np.random.normal(0, args.sigma, size)).to(device)

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": [], "value":[], "entropy":[], "reward":[]}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)

value_, reward_ = [[]], [[]]
idx = 0
prev_val = None
while log_done_counter < args.episodes:
    actions, value = agent.get_actions(obss)
    # ent = agent.get_entropy(obss)
    obss, rewards, dones, _ = env.step(actions)

    agent.analyze_feedbacks(rewards, dones)
    value_[idx].append(round(value[0], 3))
    # reward_[idx].append(rewards[0])

    log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
    log_episode_num_frames += torch.ones(args.procs, device=device)

    for i, done in enumerate(dones):
        if done:
            log_done_counter += 1
            logs["return_per_episode"].append(log_episode_return[i].item())
            logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())
            logs["value"].append(prev_val[i])
            logs["reward"].append(rewards[i])
            if i==0:
                v_t_1 = value_[idx][0]
                r_t = rewards[0]
                v_t = value_[idx][0]
                a_t = r_t + v_t_1 - v_t
                reward_[idx].append(round(r_t, 3))
                for t in range(len(value_[idx])-2, -1, -1):
                    v_t_1 = value_[idx][t+1]
                    v_t = value_[idx][t]
                    a_t = min(0.99*v_t_1 - v_t + 0.95*0.99*a_t, r_t)
                    reward_[idx].append(round(a_t, 3))
                reward_[idx].reverse()

                idx += 1
                value_.append([])
                reward_.append([])

                # print(value_[0])
                # print(reward_[0])
                # print(np.corrcoef(np.array(value_[0]), np.array(reward_[0]))[0][1])
                # exit()

    mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
    log_episode_return *= mask
    log_episode_num_frames *= mask
    prev_val = value

end_time = time.time()

corr = []
for i in range(idx):
    v, r = np.array(value_[i]), np.array(reward_[i])
    v, r = np.round((v - np.min(v)) / (np.max(v) - np.min(v)), 3), np.round((r - np.min(r)) / (1 - np.min(r)), 3)
    c = np.corrcoef(v, r)[0][1]
    # print(list(v), list(r), c)
    # print()
    if np.isnan(c):
        c = 0
    value_list.append(list(v))
    reward_list.append(list(r))
    env_list.append(args.env)
    label_list.append(args.label)
    corr_list.append(c)
    corr.append(c)
df1 = pd.DataFrame()
df1['value'], df1['reward'], df1['env'], df1['label'], df1['corr'] = value_list, reward_list, env_list, label_list, corr_list
df1.to_csv(traj_dir, index=False)
print(np.mean(corr))
# exit()

# Print logs
num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
      .format(num_frames, fps, duration, 
              *return_per_episode.values(),
              *num_frames_per_episode.values()))

# Print worst episodes

n = args.worst_episodes_to_show
if n > 0:
    print("\n{} worst episodes:".format(n))

    indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
    for i in indexes[:n]:
        print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
