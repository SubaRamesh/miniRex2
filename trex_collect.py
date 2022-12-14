from typing import List
import gym
import pickle
import time
import sys

from human import TerminalHuman
from keyboard_agent import play

env = gym.make('LunarLander-v2')

N_DEMOS = 2
SEED = 0


def run() -> List:
    # preliminary checks
    assert N_DEMOS > 0 and N_DEMOS < 10, 'N_DEMOS must be between 0 and 10'
    
    ## collect demonstrations
    demonstrations = play(env, N_DEMOS, SEED, False)
    
    ### Creating human object
    H = TerminalHuman(env, "rank")

    ## Querying human
    # ranking formatting must be (1, 2, 3)
    print('\a')
    ranking = H.input([controls for _, controls, _ in demonstrations])
    
    sorted_demonstrations = [x for _, x in sorted(zip(ranking, demonstrations))]

    return sorted_demonstrations    
    
    

if __name__ == "__main__":
    start = time.time()

    args = sys.argv
    user = args[1]
    demos = run()
    name = f"{user}-trex"
    with open(name + ".pickle", 'wb') as f:
        pickle.dump(demos, f)

    end = time.time()
    print(f"Total time taken: {end - start} s")