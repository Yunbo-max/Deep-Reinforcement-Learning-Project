from env import KukaReachVisualEnv
from RL_brain import DDPG

#Initial value
MAX_EPISODES = 500
MAX_EP_STEPS = 200

# set env
env = TotalEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)


# start training
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_r = 0.
    for j in range(MAX_EP_STEPS):
        env.render()

        a = rl.choose_action(s)

        s_, r, done = env.step(a)

        rl.store_transition(s, a, r, s_)

        if rl.memory_full:
            # start to learn once has fulfilled the memory
            rl.learn()

        s = s_