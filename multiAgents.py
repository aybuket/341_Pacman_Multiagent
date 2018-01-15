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
from searchAgents import mazeDistance
from game import Directions
import random, util, searchAgents

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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = currentGameState.getCapsules()

        "*** YOUR CODE HERE ***"

        ghost = 0
        f = 0
        for food in newFood:
            dis = manhattanDistance(newPos, food)
            f += dis
        for g in newGhostStates:
            dis = manhattanDistance(newPos,g.getPosition())
            if dis < 3:
                ghost += -99

        return successorGameState.getScore() + ghost + 1.0/(1+f)

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
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        def minvalue(state, index, depth):
            v = float('inf')
            for act in state.getLegalActions(index):
                v = min(v, value(state.generateSuccessor(index, act), index+1, depth))
            return v
        def maxvalue(state, index, depth):
            v = -float('inf')
            for act in state.getLegalActions(index):
                v = max(v, value(state.generateSuccessor(index, act), index+1, depth))
            return v

        def value(state, index, depth):
            if index == num:
                depth = depth + 1
            index = index % num
            if depth == self.depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state)
            elif index == 0:
                return maxvalue(state, index, depth)
            else:
                return minvalue(state, index, depth)

        actions = gameState.getLegalActions(0)
        maxAct = None
        v = -float('inf')
        for act in actions:
            successor = gameState.generateSuccessor(0, act)
            va = value(successor, 1, 0)
            if va > v:
                v = va
                maxAct = act
        return maxAct

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        def minvalue(state, index, depth, a, b):
            v = float('inf')
            for act in state.getLegalActions(index):
                v = min(v, value(state.generateSuccessor(index, act), index+1, depth, a, b))
                if v < a:
                    return v
                b = min(b,v)
            return v
        def maxvalue(state, index, depth, a, b):
            v = -float('inf')
            for act in state.getLegalActions(index):
                v = max(v, value(state.generateSuccessor(index, act), index+1, depth, a, b))
                if  v > b:
                    return v
                a = max(a,v)
            return v

        def value(state, index, depth, a, b):
            if index == num:
                depth = depth + 1
            index = index % num
            if depth == self.depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state)
            elif index == 0:
                return maxvalue(state, index, depth, a, b)
            else:
                return minvalue(state, index, depth, a, b)

        actions = gameState.getLegalActions(0)
        maxAct = None
        v = -float('inf')
        a = -float('inf')
        b = float('inf')
        for act in actions:
            successor = gameState.generateSuccessor(0, act)
            va = value(successor, 1, 0, a, b)
            a = max(va, a)
            if va > v:
                v = va
                maxAct = act
        return maxAct

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
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()

        def expvalue(state, index, depth):
            v = 0
            actions = state.getLegalActions(index)
            for act in actions:
                p = 1.0/len(actions)
                v = v + p * value(state.generateSuccessor(index, act), index+1, depth)
            return v

        def maxvalue(state, index, depth):
            v = -float('inf')
            for act in state.getLegalActions(index):
                v = max(v, value(state.generateSuccessor(index, act), index + 1, depth))
            return v

        def value(state, index, depth):
            if index == num:
                depth = depth + 1
            index = index % num
            if depth == self.depth or len(state.getLegalActions(index)) == 0:
                return self.evaluationFunction(state)
            elif index == 0:
                return maxvalue(state, index, depth)
            else:
                return expvalue(state, index, depth)

        actions = gameState.getLegalActions(0)
        maxAct = None
        v = -float('inf')
        for act in actions:
            successor = gameState.generateSuccessor(0, act)
            va = value(successor, 1, 0)
            if va > v:
                v = va
                maxAct = act
        return maxAct

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      I wrote very similar function to evaluation function. In the evaluation function, it just looks
      the distance to food and ghosts.
      I used same calculation for food and the ghost if the scaredtimes is 1 or 0.
      It tries to chase the ghosts if the scaredtimes is not 1 or 0.
      It is also try to eat the capsules to get more points.
      I used 1.0/(1+var) to get bigger number if it is close and smaller number if it is far.

    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    ghost = 0
    cap = float('inf')
    f = 0
    scaredGhost = 0
    for food in foods:
        dis = manhattanDistance(position, food)
        f += dis
    for caps in capsules:
        dis = manhattanDistance(position, caps)
        cap = min(cap, dis)
    for g in ghostStates:
        dis = manhattanDistance(position, g.getPosition())
        if dis < 4:
            ghost += -99
        scaredGhost += dis
    if scaredTimes < 2:
        g = ghost
    else:
        g = 1.0 / (1 + scaredGhost)
    return currentGameState.getScore() + g + 1.0 / (1 + cap) + 1.0 / (1 + f)

# Abbreviation
better = betterEvaluationFunction

