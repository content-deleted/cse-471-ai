# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    Some notes:
    - Get start node: problem.getStartState()
    - Get successor: problem.getSuccessors( tuple (x,y) )
        -returns array of tuples.isGoalState(problem.getStartState())
            return 
        -[((5, 4), 'South', 1), ((4, 5), 'West', 1)]
        -for each element in array ( (tuple x,y) , action to reach , cost ) 
    - Check if node is goal: problem.isGoalState( tuple (x,y) )
    """
    fringe = util.Stack();
    return GraphSearch(problem, fringe);

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue();
    return GraphSearch(problem, fringe);

def GraphSearch(problem, fringe):
    fringe.push( ((problem.getStartState(),'',0), 0) );
    closed = [];

    while(not fringe.isEmpty()): 
        node = fringe.pop();

        unique = node[0][0] not in closed;
        if(unique): 
            closed.append(node[0][0]);
            if (problem.isGoalState(node[0][0])): 
                sol = [node[0][1]];
                while(node[1]):
                    sol.insert(0, node[1][0][1]);
                    node = node[1];
                del sol[0];
                return sol;
            map(lambda x: fringe.push( (x, node) ), problem.getSuccessors(node[0][0]));

    return [];


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    fringe = util.PriorityQueue();

    fringe.push( ((problem.getStartState(),'',0), 0 , 0) , 0 );
    closed = [];

    while(not fringe.isEmpty()): 
        node = fringe.pop();
        unique = node[0][0] not in closed;
        if(unique): 
            closed.append(node[0][0]);
            if (problem.isGoalState(node[0][0])): 
                sol = [node[0][1]];
                while(node[1]):
                    sol.insert(0, node[1][0][1]);
                    node = node[1];
                del sol[0];
                return sol;
            map(lambda x: fringe.push( (x, node, node[2] + x[2]), (x[2] + node[2]) ), problem.getSuccessors(node[0][0]));

    return [];

def nullHeuristic(state, problem):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue();

    fringe.push( ((problem.getStartState(),'',0), 0 , 0) , 0 );
    closed = [];

    while(not fringe.isEmpty()): 
        node = fringe.pop();

        unique = node[0][0] not in closed;
        if(unique): 
            closed.append(node[0][0]);

            if (problem.isGoalState(node[0][0])): 
                sol = [node[0][1]];
                while(node[1]):
                    sol.insert(0, node[1][0][1]);
                    node = node[1];
                del sol[0];
                return sol;
            suc = problem.getSuccessors(node[0][0]);
            
            map(lambda x: fringe.push( (x, node, node[2] + x[2]), heuristic(x[0], problem ) + (x[2] + node[2]) ), suc);

    return [];


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
