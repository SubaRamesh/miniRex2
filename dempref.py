import time, sys
sys.path.insert(0,'../..')

import numpy as np
import pandas as pd
import pickle

import utils
import gym
import human
import query_generation
import sampling
import glob

env = gym.make('LunarLander-v2')
#timesteps not included yet

### Parameters
N_QUERY = 2
UPDATE_FUNC = "approx"
QUERY_LENGTH = 10
INC_PREV_QUERY = False
GEN_SCENARIO = False
N_PREF_ITERS = 15
EPSILON = 0

VARIANCE_THRESH = 0

N_SAMPLES_SUMM = 50000
N_SAMPLES_EXP = N_SAMPLES_SUMM

BETA_DEMO = 0.1
BETA_PREF = 5


def run() -> pd.DataFrame:
    ### Creating data frame to store data in
    # pref_iter correponds to the iteration of the preference loop in the particular run
    # run is the type of data being stored; options are "mean", "var", "m"
    # value is the actual value being stored
    df = pd.DataFrame(columns=["pref_iter", "type", "value"])
    
    ### Creating sampler 
    # dim_features could be 4
    sampler = sampling.Sampler(n_query=N_QUERY, dim_features=4, update_func=UPDATE_FUNC,
                               beta_demo=BETA_DEMO, beta_pref=BETA_PREF)
    ### Creating query generator
    qg = query_generation.ApproxQueryGenerator(env=env, num_queries=N_QUERY, query_length=QUERY_LENGTH,
                                               num_expectation_samples=N_SAMPLES_EXP,
                                               include_previous_query=INC_PREV_QUERY,
                                               generate_scenario=GEN_SCENARIO, update_func=UPDATE_FUNC,
                                               beta_pref=BETA_PREF)

    ### Creating human object
    H = human.TerminalHuman(env, UPDATE_FUNC)
    ### Collecting and loading demonstrations
    # demo_names = DOM.collect_dems()
    demo_path = 'data/dem_pref_demos/'
    demo_names = glob.glob(f"{demo_path}/gym-*.pickle")
    demos = [pickle.load(open(demo_name, 'rb'), encoding='latin1')[1] for demo_name in demo_names]
    # demos = [DOM.fetch_to_mujoco(x) for x in demos]
    # if INC_PREV_QUERY:
    #     last_query_picked = demos[0]
    phi_demos = [utils.np_features(x, dempref=True) for x in demos]
    sampler.load_demo(np.array(phi_demos))
    # print(f"Phi Demos at init: {sampler.phi_demos}")

    ### Computing initial estimates
    samples = sampler.sample(N=N_SAMPLES_SUMM)
    # print(f"Samples: {samples}\n")
    mean_w = np.mean(samples, axis=0)
    mean_w = mean_w / np.linalg.norm(mean_w)
    var_w = np.var(samples, axis=0)
    print('Mean: ' + str(mean_w))
    print('Var: ' + str(sum(var_w)))
    data = [[0, "mean", mean_w],
            [0, "var", var_w]]
    df = df.append(pd.DataFrame(data, columns=["pref_iter", "type", "value"]), ignore_index=True)
    if sum(var_w) < VARIANCE_THRESH:
        print("Variance is now below threshold; EXITING.")
        return df

    ### Preferences loop
    for j in range(N_PREF_ITERS):
        print("\n\n*** Preferences # %d\n" % (j + 1))

        ## Generate queries
        # if INC_PREV_QUERY:
        #     queries = qg.queries(samples, last_query_picked)
        # else:
        queries, query_controls = qg.queries(samples)
        # mujoco_queries = [DOM.mujoco_to_fetch(x) for x in queries]

        ## Querying human
        print('\a')
        best = H.input(query_controls, dempref=True)
        # if INC_PREV_QUERY:
        #     last_query_picked = queries[best]

        ## Creating dictionary mapping rankings to features of queries and loading into sampler
        features = [utils.np_features(x, dempref=True) for x in queries]
        phi = {k: features[k] for k in range(len(queries))}
        sampler.load_prefs(phi, best)

        ## Recording data from this run
        samples = sampler.sample(N=N_SAMPLES_SUMM)
        mean_w = np.mean(samples, axis=0)
        mean_w = mean_w / np.linalg.norm(mean_w)
        var_w = np.var(samples, axis=0)
        print('Mean: ' + str(mean_w))
        print('Var: ' + str(sum(var_w)))
        data = [[j+1, "mean", mean_w],
                [j+1, "var", var_w]]
        df = df.append(pd.DataFrame(data, columns=["pref_iter", "type", "value"]), ignore_index=True)
        if sum(var_w) < VARIANCE_THRESH:
            print("Variance is now below threshold; EXITING.")
            return df

    return df


if __name__ == "__main__":
    start = time.time()

    args = sys.argv
    user = args[1]
    df = run()
    name = f"{user}-dempref"
    with open(name + ".pickle", 'wb') as f:
        pickle.dump(df, f)

    end = time.time()
    print(f"Total time taken: {end - start} s")

