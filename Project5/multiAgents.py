# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        curFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        closestGhostDistance = 99999
        for ghost in newGhostStates:
            dist = util.manhattanDistance(ghost.getPosition(), newPos)
            if dist < closestGhostDistance: closestGhostDistance = dist

        scoreDiff = successorGameState.getScore() - currentGameState.getScore()

        if(curFood[newPos[0]][newPos[1]]):
            closestFruitDistance = 0.5
        else:
            closestFruitDistance = 99999
            for pos in curFood.asList():
                if(newFood[pos[0]][pos[1]]):
                    dist = util.manhattanDistance(pos, newPos)
                    if dist < closestFruitDistance: closestFruitDistance = dist
        
        scoreEval = -10.0 / (closestGhostDistance * closestGhostDistance) + 5.0 / closestFruitDistance
        
        return -1000 if (closestGhostDistance <= 2) else scoreEval

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        def minimax(currentDepth, currentGameState, agentNum):
            if(currentDepth == self.depth): return self.evaluationFunction(currentGameState)
            
            actions = currentGameState.getLegalActions(agentNum)
            newAgentNum = agentNum + 1
            
            if(newAgentNum >= currentGameState.getNumAgents()): 
                newAgentNum = 0
                currentDepth+=1
            
            successors = map(lambda action: minimax(currentDepth, currentGameState.generateSuccessor(agentNum, action),newAgentNum), actions)

            if not successors:
                return self.evaluationFunction(currentGameState)
            else:
                return max(successors) if agentNum == 0 else min(successors)
    
        #if(self.depth != 0):
        actions = gameState.getLegalActions(0)
        bestAction = None
        Max = -10000000
        startingDepth = 0
        for action in actions:
            v = minimax(startingDepth,  gameState.generateSuccessor(0, action), 1)
            if(v > Max):
                Max = v
                bestAction = action
        return bestAction
            
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(currentDepth, currentGameState, agentNum, a, B):
            if(currentDepth == self.depth): return self.evaluationFunction(currentGameState)
            
            actions = currentGameState.getLegalActions(agentNum)
            newAgentNum = agentNum + 1
            
            if(newAgentNum >= currentGameState.getNumAgents()): 
                newAgentNum = 0
                currentDepth+=1

            successors = list()
            for action in actions:
                successor = alphaBeta(currentDepth, currentGameState.generateSuccessor(agentNum, action),newAgentNum, a, B)
                successors.append(successor)
                
                if(agentNum == 0):
                    a = max(a, successor)
                else:
                    B = min(B, successor)
                 
                if( B < a ): break

            if not successors:
                return self.evaluationFunction(currentGameState)
            else:
                return max(successors) if agentNum == 0 else min(successors)
    
        #if(self.depth != 0):
        actions = gameState.getLegalActions(0)
        bestAction = None
        Max = -10000000
        startingDepth = 0
        a = Max
        B = 10000000
        for action in actions:
            v = alphaBeta(startingDepth,  gameState.generateSuccessor(0, action), 1, a, B)
            if(v > Max):
                Max = a = v
                bestAction = action

        return bestAction
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        def expectimax(currentDepth, currentGameState, agentNum):
            if(currentDepth == self.depth): return self.evaluationFunction(currentGameState)
            
            actions = currentGameState.getLegalActions(agentNum)
            newAgentNum = agentNum + 1
            
            if(newAgentNum >= currentGameState.getNumAgents()): 
                newAgentNum = 0
                currentDepth+=1
            
            successors = map(lambda action: expectimax(currentDepth, currentGameState.generateSuccessor(agentNum, action),newAgentNum), actions)

            if not successors:
                return self.evaluationFunction(currentGameState)
            else:
                return max(successors) if agentNum == 0 else sum(successors) / float(len(successors))
    
        #if(self.depth != 0):
        actions = gameState.getLegalActions(0)
        bestAction = None
        Max = -10000000
        startingDepth = 0
        for action in actions:
            v = expectimax(startingDepth,  gameState.generateSuccessor(0, action), 1)
            if(v > Max):
                Max = v
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    curPos = currentGameState.getPacmanPosition()
    curFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    

    closestGhostDistance = 100
    for ghost in ghostStates:
        if(ghost.scaredTimer > 2): continue
        dist = util.manhattanDistance(ghost.getPosition(), curPos)
        if dist < closestGhostDistance: closestGhostDistance = dist

    if(closestGhostDistance == 100): closestGhostDistance = -0.5

    if(closestGhostDistance==0): return -1000000

    score = currentGameState.getScore()

    closestFruitDistance = 99999
    for pos in curFood.asList():
        if(curFood[pos[0]][pos[1]]):
            dist = util.manhattanDistance(pos, curPos)
            if dist < closestFruitDistance: closestFruitDistance = dist 

    capsuleCount = len(currentGameState.getCapsules())

    scoreEval = currentGameState.getScore() - 1.0 / (closestGhostDistance) + 5.0 / closestFruitDistance - capsuleCount

    return -10000 * closestGhostDistance if (closestGhostDistance <= 2) else scoreEval

# Abbreviation
better = betterEvaluationFunction

