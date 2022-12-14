import gym
import numpy as np
import time


def np_features(t, dempref = False):
    # distance from landing pad at (0, 0)
    # weight should be negative
    # print('getting np_features')
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
    
    if dempref:
        phi_total = list(np.mean(lst_of_features, axis=0))
        # phi_total.append(np_path_length(t))
        phi_total.append(np_final_position(t))
        
        return np.array(phi_total)
    return lst_of_features


def watch(env, controls, seed: int = None, render:bool=True):
    if len(controls) > 0:
        start = time.time()
        # mapping = {1: [0, -1], 2: [1, 0], 3: [0, 1], 0: [0, 0]}
        # controls_ = []
        # for i in range(len(controls)):
        #     controls_.append(mapping[controls[i][0]])
        
        features = watch2(env, controls, seed, render)
        # print(reward)
        end = time.time()
        # print("Finished running game in " + str(end - start) + "s")
        return features
    return []
    
def watch2(env, controls, seed:int=None, render:bool=True):
    env.seed(seed)
    env.reset()
    features = []
    reward = 0
    
    for i in range(len(controls)):
        if render:
            frame_delay_ms = 0.1
            time.sleep(frame_delay_ms/1000)
            env.render()
        a = controls[i]#[0]
        a = int(np.floor(a))
        # print(a)
        try:
            obser, r, done, info = env.step(a)
            features.append(obser)
            reward += r
            
            
            if done: # quit if game over
                print(f"{i}: {reward}")
                break
        except:
            continue
            # print(f"invalid action {a} --> {int(np.floor(a))}")
        #print(self.state)
            
    env.close()
    
    return features

def dempref_run(env: gym.Env,  controls: np.ndarray, time_steps: int = 150, render: bool = False, seed: int = 0):
    control_size = 1
    c = np.array([[0.] * control_size] * time_steps)
    num_intervals = len(controls)//control_size # 8
    interval_length = time_steps//num_intervals # 18

    assert interval_length * num_intervals == time_steps, f"Number of generated controls: {interval_length}, {num_intervals} must be divisible by total time steps: {time_steps}."

    j = 0
    for i in range(num_intervals):
        c[i * interval_length: (i + 1) * interval_length] = [controls[j + i] for i in range(control_size)]
        j += control_size

    print(f"controls ({len(c)}): \n{[int(np.floor(x))for x in c]}")

    env.seed(seed)
    obser = env.reset()
    s = [obser]
    reward = 0
    for i in range(time_steps):
        try:
            action = int(np.floor(c[i]))
            if action < 0:
                action = 0
            elif action > 3:
                action = 3
            
            results = env.step(action)
            if render:
                frame_delay_ms = 0
                time.sleep(frame_delay_ms/1000)
                env.render()
        except:
            print(f"Caught unstable simulation; skipping. Last control: {int(np.floor(c[i]))}")
            return (None, None)
            # return (None, None)
        
        obser = results[0]
        reward += results[1]
        s.append(obser)
        if results[2]:
            break
    if len(s) <= time_steps:
        c = c[:len(s), :]
    else:
        c = np.append(c, [np.zeros(control_size)], axis=0)
        
    print(f"reward: {reward}\n")
    return (s, np.array([c]))

    