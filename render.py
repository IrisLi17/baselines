import tensorflow as tf
import gym
from baselines import deepq
import baselines.common.tf_util as U 
from baselines.common import set_global_seeds
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari
from baselines import bench
import numpy as np
from baselines.deepq.simple import ActWrapper

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--model-dir', type=str, default=None)
    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    sess = tf.Session()
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=model,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        gamma=1.0,
        grad_norm_clipping=10,
        param_noise=False
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': model,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    U.initialize()

    # model_file = "/home/liyunfei/Projects/baselines/model/model"
    if args.model_dir is None:
        model_file = os.path.join("./model/", args.env, "steps"+str(args.num_timesteps))
    else:
        model_file = os.path.join("./model/", args.env, args.model_dir)
    U.load_state(model_file)

    obs = env.reset()
    done = 0
    while not done:
        env.render()
        kwargs = {}
        action = act(np.array(obs)[None], update_eps=0, **kwargs)[0]
        print(action)
        obs,rew,done,_ = env.step(action)

    env.close()

if __name__ == "__main__":
    main()