''' ReinforceAgentClass.py: Class for a REINFORCE agent, from:

    Williams, Ronald J. "Simple statistical gradient-following algorithms for
    connectionist reinforcement learning." Machine learning 8.3-4 (1992): 229-256.
    
Note: This agent has not yet been implemented.
'''

# Python imports.
from collections import defaultdict

# Other imports
from simple_rl.agents.PolicyGradientAgentClass import PolicyGradientAgent

class ReinforceAgent(PolicyGradientAgent):
    ''' Class for REINFORCE agent, '''

    def __init__(self, actions, name=""):
        raise NotImplementedError("REINFORCE not yet implemented.")

    def policy(self, state):
        '''
        Args:
        state (simple_rl.State)

        Returns:
        	(str)
        '''
        # Sample from
        return self.actions[np.random.multinomial(1, self.pmf_a_given_s[state].values()).tolist().index(1)]


    def act(self, state, reward):
        '''
        Args:
            state (State)
            reward (float)

        Returns:
            (str)
        '''
        pass

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)

        Summary:
            Perform a state of policy gradient.
        '''
        pass
