#!/usr/bin/env python
from __future__ import print_function

import sys, gym, time
import pickle
import random

#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

env = gym.make('LunarLander-v2')
# env = gym.make('SpaceInvaders-v0')

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()

    states = [obser]
    controls = []

    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        controls.append([a])

        obser, r, done, info = env.step(a)
        states.append(obser)

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open == False or done:
            controls.append([0])
            env.close()
            return (False, states, controls, total_reward)
        # if done: break
        # if human_wants_restart: break
        # while human_sets_pause:
        #     env.render()
        #     time.sleep(0.1)
        time.sleep(0.01)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
    return states, controls, total_reward


def play(env, n_runs, seed, save=False):
    names = []

    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    for i in range(n_runs):
        env.seed(seed)
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        window_still_open, states, controls, total_reward = rollout(env)
        if save:
            name = str(time.time())
            with open('data/demonstrations/gym-' + str(name) + ".pickle", 'wb') as f:
                pickle.dump((controls, states, total_reward), f)
            names.append(name)
        else:
            names.append(states, controls, total_reward)
    
    return names


def play_rand(env, n_runs, seed, save=False):
    # import pdb; pdb.set_trace()
    result = []
    env.seed(seed)
    env.render()
    
    print(f'running random agent for {n_runs}')
    
    for i in range(n_runs):
        obser = env.reset()

        states = [obser]
        controls = []

        skip = 0
        total_reward = 0
        total_timesteps = 0
        while 1:
            a = random.sample([i for i in range(ACTIONS)], k = 1)[0]
            total_timesteps += 1
            # print(a)
            controls.append([a])

            obser, r, done, info = env.step(a)
            states.append(obser)

            # if r != 0:
            #     print("reward %0.3f" % r)
            total_reward += r
            window_still_open = env.render()
            if window_still_open == False or done:
                controls.append([0])
                env.close()
                break
                # return (False, states, controls)
            # if done: break
            time.sleep(20/1000)
        print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))
        
        if save:
            name = str(time.time())
            with open('data-rand/demonstrations/gym-' + str(name) + ".pickle", 'wb') as f:
                pickle.dump((controls, states, total_reward), f)
            result.append(name)
        else:
            result.append((states, controls, total_reward))

    return result
