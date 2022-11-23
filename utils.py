import gym
import numpy as np
import time


def np_features(t):
    # distance from landing pad at (0, 0)
    # weight should be negative
    # import pdb; pdb.set_trace()
    def np_dist_from_landing_pad(x):
        return -15*np.exp(-np.sqrt(x[0]**2+x[1]**2))

    # angle of lander
    # angle is 0 when upright (positive in left direction, negative in right)
    # weight should be positive
    def np_lander_angle(x):
        return 15*np.exp(-np.abs(x[4]))

    # velocity of lander
    # weight should be negative
    def np_velocity(x):
        return -10*np.exp(-np.sqrt(x[2]**2+x[3]**2))

    # total path length
    # weight should be positive
    def np_path_length(t):
        states = t
        total = 0
        for i in range(1, len(states)):
            total += np.sqrt((states[i][0] - states[i-1][0])**2 + (states[i][1] - states[i-1][1])**2)
        total = np.exp(-total)
        return 10 * total
    
    # final position
    # weight should be negative
    def np_final_position(t):
        x = t[-1]
        return -30*np.exp(-np.sqrt(x[0] ** 2 + x[1] ** 2))

    lst_of_features = []
    for i in range(len(t)):
        # import pdb; pdb.set_trace()
        x = t[i]
        if i > len(t)//5:
            phi = np.stack([
                np_dist_from_landing_pad(x),
                np_lander_angle(x),
                np_velocity(x),
            ])
        else:
            phi = np.stack([
                np_dist_from_landing_pad(x),
                np_lander_angle(x),
                0,
            ])
        lst_of_features.append(phi)
        # phi_total = list(np.mean(lst_of_features, axis=0))
        # # phi_total.append(np_path_length(t))
        # phi_total.append(np_final_position(t))
        # return np.array(phi_total)
    return lst_of_features


def watch(env, controls, seed: int = None):
    if len(controls) > 0:
        # mapping = {1: [0, -1], 2: [1, 0], 3: [0, 1], 0: [0, 0]}
        # controls_ = []
        # for i in range(len(controls)):
        #     controls_.append(mapping[controls[i][0]])
        
        watch2(env, controls, seed)
    
def watch2(env, controls, seed:int=None, on_real_robot:bool=False):
    env.seed(seed)
    env.reset()
    
    for i in range(len(controls)):
        env.render()
        a = controls[i][0]
        print(a)
        results = env.step(a)
        # print(f'{results[1]:.3f}')
        #print(self.state)
        frame_delay_ms = 20
        time.sleep(frame_delay_ms/1000)\
            
        if results[2]: # quit if game over
            break
    env.close()