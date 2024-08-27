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


import random
from game import Agent
import util
from util import manhattanDistance

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Evaluates a state-action pair to help Pacman decide the best action.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Calculate the distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(foodDistances) > 0:
            closestFoodDistance = min(foodDistances)
        else:
            closestFoodDistance = 1

        # Calculate the distance to the nearest ghost
        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        if len(ghostDistances) > 0:
            closestGhostDistance = min(ghostDistances)
        else:
            closestGhostDistance = 1

        # Use the reciprocal of distances as features for the evaluation function
        score = successorGameState.getScore() + 1.0 / closestFoodDistance - 2.0 / closestGhostDistance

        return score


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
        """
        action, _ = self.minimax(0, 0, gameState)
        return action

    def minimax(self, curr_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agent_index)
        best_action = None

        if agent_index == 0:  # Pacman's turn (Maximizing player)
            best_score = float('-inf')
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, nextState)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action, best_score

        else:  # Ghost's turn (Minimizing player)
            best_score = float('inf')
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, nextState)
                if score < best_score:
                    best_score = score
                    best_action = action
            return best_action, best_score

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction with alpha-beta pruning.
        """
        action, _ = self.alphabeta(0, 0, gameState, float('-inf'), float('inf'))
        return action

    def alphabeta(self, curr_depth, agent_index, gameState, alpha, beta):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agent_index)
        best_action = None

        if agent_index == 0:  # Pacman's turn (Maximizing player)
            best_score = float('-inf')
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agent_index, action)
                _, score = self.alphabeta(curr_depth, agent_index + 1, nextState, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_action, best_score

        else:  # Ghost's turn (Minimizing player)
            best_score = float('inf')
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agent_index, action)
                _, score = self.alphabeta(curr_depth, agent_index + 1, nextState, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_action, best_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction.
        """
        action, _ = self.expectimax(0, 0, gameState)
        return action

    def expectimax(self, curr_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agent_index)
        best_action = None

        if agent_index == 0:  # Pacman's turn (Maximizing player)
            best_score = float('-inf')
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, nextState)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action, best_score

        else:  # Ghost's turn (Expectimax, taking the average of all moves)
            avg_score = 0
            for action in legalMoves:
                nextState = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, nextState)
                avg_score += score
            avg_score /= len(legalMoves)
            return None, avg_score


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function (question 5).
    """
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Distance to the nearest food
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food]
    closestFoodDistance = min(foodDistances) if foodDistances else 1

    # Distance to the nearest ghost
    ghostDistances = [manhattanDistance(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates]
    closestGhostDistance = min(ghostDistances) if ghostDistances else 1

    # Calculate score with food and ghost distances
    score = currentGameState.getScore() + 1.0 / closestFoodDistance - 2.0 / closestGhostDistance

    return score



# Abbreviation
better = betterEvaluationFunction

