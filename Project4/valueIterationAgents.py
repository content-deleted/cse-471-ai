# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here

        # intilize values
        # if the state is a terminal state we want to set its V to it's exit reward
        # else we init to zero 
        for state in self.mdp.getStates():
            # mdp.getReward(state) if mdp.isTerminal(state) else 0
            self.values[state] = 0
        
        '''
        for state in self.mdp.getStates():
                maxA = 0
                for action in self.mdp.getPossibleActions(state):
                    state2AndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

                    for stateAndProb in state2AndProbs:
                        #print "state: ", state, " action: ", action, "next state: ", stateAndProb[0]
                        reward = self.mdp.getReward(state, action, stateAndProb[0])
                        self.values[state] = self.discount * ( reward * stateAndProb[1] )
                        #print " REWARD:", reward
        '''
        
        for num in xrange(0, self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                ourSums = list()
                if mdp.isTerminal(state): 
                    continue
                for action in self.mdp.getPossibleActions(state):
                    stateProbSum = self.computeQValueFromValues(state, action)
                    ourSums.append( stateProbSum )
                newValues[state] = max(ourSums)
            self.values = newValues
                        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        state2AndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        stateProbSum = 0

        for stateAndProb in state2AndProbs:
            statePrime, Prob = stateAndProb
            reward = self.mdp.getReward(state, action, statePrime)
            stateProbSum += Prob * ( reward + self.discount * self.values[statePrime]  )

        return stateProbSum

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # Check for terminal state
        if self.mdp.isTerminal(state): return None
            
        # return the action with the highest V
        return self.values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
