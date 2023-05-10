# Python imports.
import sys
import logging

import srl_example_setup
from simple_rl.agents import RandomAgent, QLearningAgent
from simple_rl.tasks import GymMDP
from simple_rl.run_experiments import run_agents_on_mdp
import minigrid
from minigrid.wrappers import FullyObsWrapper

def main(open_plot=True):
    # Gym MDP
    gym_mdp = GymMDP(env_name='MiniGrid-Empty-5x5-v0', render=False, wrapper=FullyObsWrapper)
    num_feats = gym_mdp.get_num_state_feats()

    # Setup agents and run.
    rand_agent = RandomAgent(gym_mdp.get_actions())
    q_agent = QLearningAgent(gym_mdp.get_actions())
    # lin_q_agent = LinearQAgent(gym_mdp.get_actions(), num_feats)
    run_agents_on_mdp([q_agent, rand_agent], gym_mdp, instances=5, episodes=100, steps=200, open_plot=open_plot, verbose=False,
                      cumulative_plot=True, track_disc_reward=False)



if __name__ == "__main__":
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.ERROR)
    main(open_plot=not sys.argv[-1] == "no_plot")
