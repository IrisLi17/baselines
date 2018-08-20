import gym

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari
import numpy as np
import os
import datetime
import re

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--expert',type=int, default=0)
    parser.add_argument('--pre-timesteps',type=int, default=int(1e4))
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default=None)
    args = parser.parse_args()
    pattern1 = re.compile('SpaceInvaders')
    pattern2 = re.compile('MsPacman')
    pattern3 = re.compile('Qbert')
    pattern4 = re.compile('VideoPinball')
    pattern5 = re.compile('MontezumaRevenge')
    assert pattern1.match(args.env) or pattern2.match(args.env) or pattern3.match(args.env) or pattern4.match(args.env) or pattern5.match(args.env)
    if pattern1.match(args.env):
        g = 'spaceinvaders'
    elif pattern2.match(args.env):
        g = 'mspacman'
    elif pattern3.match(args.env):
        g = 'qbert'
    elif pattern4.match(args.env):
        g = 'pinball'
    elif pattern5.match(args.env):
        g = 'revenge'
    if args.log_dir is None:
        dir = os.path.join('./logs/', args.env, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    else:
        dir = os.path.join('./logs/', args.env, args.log_dir)
    if args.model_dir is None:
        model_dir = os.path.join('./model/', args.env, "steps"+str(args.num_timesteps))
    else:
        model_dir = os.path.join('./model/', args.env, args.model_dir)
    logger.configure(dir = dir)
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
        layer_norm=True
    )
    act = deepq.learn(
        env,
        g,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        use_expert=bool(args.expert),
        pre_timesteps=args.pre_timesteps,
        model_file=model_dir
    )
    # act.save("pong_model.pkl") XXX
    obs = env.reset()
    done = 0
    while not done:
        env.render()
        kwargs = {}
        action = act(np.array(obs)[None], update_eps=0, **kwargs)[0]
        print(action)
        obs,rew,done,_ = env.step(action)

    env.close()


if __name__ == '__main__':
    main()
