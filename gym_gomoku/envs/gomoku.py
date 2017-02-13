import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys, os
import six

from gym_gomoku.envs.util import gomoku_util
from gym_gomoku.envs.util import make_random_policy
from gym_gomoku.envs.util import make_beginner_policy
from gym_gomoku.envs.util import make_medium_policy
from gym_gomoku.envs.util import make_expert_policy

# Rules from Wikipedia: Gomoku is an abstract strategy board game, Gobang or Five in a Row, it is traditionally played with Go pieces (black and white stones) on a go board with 19x19 or (15x15) 
# The winner is the first player to get an unbroken row of five stones horizontally, vertically, or diagonally. (so-calle five-in-a row)
# Black plays first if white did not win in the previous game, and players alternate in placing a stone of their color on an empty intersection.

class GomokuState(object):
    '''
    Similar to Go game, Gomoku state consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is to place stone on empty intersection
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in ['black', 'white'], 'Invalid player color'
        self.board, self.color = board, color
    
    def act(self, action):
        '''
        Executes an action for the current player
        
        Returns:
            a new GomokuState with the new board and the player switched
        '''
        return GomokuState(self.board.play(action, self.color), gomoku_util.other_color(self.color))
    
    def __repr__(self):
        '''stream of board shape output'''
        # To Do: Output shape * * * o o
        return 'To play: {}\n{}'.format(six.u(self.color), self.board.__repr__())

# Sampling without replacement Wrapper 
# sample() method will only sample from valid spaces
class DiscreteWrapper(spaces.Discrete):
    def __init__(self, n):
        self.n = n
        self.valid_spaces = list(range(n))
    
    def sample(self):
        '''Only sample from the remaining valid spaces
        '''
        if len(self.valid_spaces) == 0:
            print ("Space is empty")
            return None
        np_random, _ = seeding.np_random()
        randint = np_random.randint(len(self.valid_spaces))
        return self.valid_spaces[randint]
    
    def remove(self, s):
        '''Remove space s from the valid spaces
        '''
        if s is None:
            return
        if s in self.valid_spaces:
            self.valid_spaces.remove(s)
        else:
            print ("space %d is not in valid spaces" % s)


### Environment
class GomokuEnv(gym.Env):
    '''
    GomokuEnv environment. Play against a fixed opponent.
    '''
    metadata = {"render.modes": ["human", "ansi"]}
    
    def __init__(self, player_color, opponent, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: Name of the opponent policy, e.g. random, beginner, medium, expert
            board_size: board_size of the board to use
        """
        self.board_size = board_size
        self.player_color = player_color
        
        self._seed()
        
        # opponent
        self.opponent_policy = None
        self.opponent = opponent
        
        # Observation space on board
        shape = (self.board_size, self.board_size) # board_size * board_size
        self.observation_space = spaces.Box(np.zeros(shape), np.ones(shape))
        
        # One action for each board position
        self.action_space = DiscreteWrapper(self.board_size**2)
        
        # Keep track of the moves
        self.moves = []
        
        # Empty State
        self.state = None
        
        # reset the board during initialization
        self._reset()
    
    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]
    
    def _reset(self):
        self.state = GomokuState(Board(self.board_size), gomoku_util.BLACK) # Black Plays First
        self._reset_opponent(self.state.board) # (re-initialize) the opponent,
        self.moves = []
        
        # Let the opponent play if it's not the agent's turn, there is no resign in Gomoku
        if self.state.color != self.player_color:
            self.state, _ = self._exec_opponent_play(self.state, None, None)
            opponent_action_coord = self.state.board.last_coord
            self.moves.append(opponent_action_coord)
        
        # We should be back to the agent color
        assert self.state.color == self.player_color
        
        # reset action_space
        self.action_space = DiscreteWrapper(self.board_size**2)
        
        self.done = self.state.board.is_terminal()
        return self.state.board.encode()
    
    def _close(self):
        self.opponent_policy = None
        self.state = None
    
    def _render(self, mode="human", close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(repr(self.state) + '\n')
        return outfile
    
    def _step(self, action):
        '''
        Args: 
            action: int
        Return: 
            observation: board encoding, 
            reward: reward of the game, 
            done: boolean, 
            info: state dict
        Raise:
            Illegal Move action, basically the position on board is not empty
        '''
        assert self.state.color == self.player_color # it's the player's turn
        
        # If already terminal, then don't do anything
        if self.done:
            return self.state.board.encode(), 0., True, {'state': self.state}
        
        # Player play
        prev_state = self.state
        self.state = self.state.act(action)
        self.moves.append(self.state.board.last_coord)
        self.action_space.remove(action) # remove current action from action_space
        
        # Opponent play
        if not self.state.board.is_terminal():
            self.state, opponent_action = self._exec_opponent_play(self.state, prev_state, action)
            self.moves.append(self.state.board.last_coord)
            self.action_space.remove(opponent_action)   # remove opponent action from action_space
            # After opponent play, we should be back to the original color
            assert self.state.color == self.player_color
        
        # Reward: if nonterminal, there is no 5 in a row, then the reward is 0
        if not self.state.board.is_terminal():
            self.done = False
            return self.state.board.encode(), 0., False, {'state': self.state}
        
        # We're in a terminal state. Reward is 1 if won, -1 if lost
        assert self.state.board.is_terminal(), 'The game is terminal'
        self.done = True
        
        # Check Fianl wins
        exist, win_color = gomoku_util.check_five_in_row(self.state.board.board_state) # 'empty', 'black', 'white'
        reward = 0.
        if win_color == "empty": # draw
            reward = 0.
        else:
            player_wins = (self.player_color == win_color) # check if player_color is the win_color
            reward = 1. if player_wins else -1.
        return self.state.board.encode(), reward, True, {'state': self.state}
    
    def _exec_opponent_play(self, curr_state, prev_state, prev_action):
        '''There is no resign in gomoku'''
        assert curr_state.color != self.player_color
        opponent_action = self.opponent_policy(curr_state, prev_state, prev_action)
        return curr_state.act(opponent_action), opponent_action
    
    @property
    def _state(self):
        return self.state
    
    @property
    def _moves(self):
        return self.moves
    
    def _reset_opponent(self, board):
        if self.opponent == 'random':
            self.opponent_policy = make_random_policy(self.np_random)
        elif self.opponent == 'beginner':
            self.opponent_policy = make_beginner_policy(self.np_random)
        elif self.opponent == 'medium':
            self.opponent_policy = make_medium_policy(self.np_random)
        elif self.opponent == 'expert':
            self.opponent_policy = make_expert_policy(self.np_random)
        else:
            raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

class Board(object):
    '''
    Basic Implementation of a Go Board, natural action are int [0,board_size**2)
    '''
    
    def __init__(self, board_size):
        self.size = board_size
        self.board_state = [[gomoku_util.color_dict['empty']] * board_size for i in range(board_size)] # initialize board states to empty
        self.move = 0                 # how many move has been made
        self.last_coord = (-1,-1)     # last action coord
        self.last_action = None       # last action made
    
    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.size + j # action index
        return a
    
    def action_to_coord(self, a):
        coord = (a // self.size, a % self.size)
        return coord
    
    def get_legal_move(self):
        ''' Get all the next legal move, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_move.append((i, j))
        return legal_move
    
    def get_legal_action(self):
        ''' Get all the next legal action, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_action = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_action.append(self.coord_to_action(i, j))
        return legal_action
    
    def copy(self, board_state):
        '''update board_state of current board values from input 2D list
        '''
        input_size_x = len(board_state)
        input_size_y = len(board_state[0])
        assert input_size_x == input_size_y, 'input board_state two axises size mismatch'
        assert len(self.board_state) == input_size_x, 'input board_state size mismatch'
        for i in range(self.size):
            for j in range(self.size):
                self.board_state[i][j] = board_state[i][j]
    
    def play(self, action, color):
        '''
            Args: input action, current player color
            Return: new copy of board object
        '''
        b = Board(self.size)
        b.copy(self.board_state) # create a board copy of current board_state
        b.move = self.move
        
        coord = self.action_to_coord(action)
        # check if it's legal move
        if (b.board_state[coord[0]][coord[1]] != 0): # the action coordinate is not empty
            raise error.Error("Action is illegal, position [%d, %d] on board is not empty" % ((coord[0]+1),(coord[1]+1)))
        
        b.board_state[coord[0]][coord[1]] = gomoku_util.color_dict[color]
        b.move += 1 # move counter add 1
        b.last_coord = coord # save last coordinate
        b.last_action = action
        return b
    
    def is_terminal(self):
        exist, color = gomoku_util.check_five_in_row(self.board_state)
        is_full = gomoku_util.check_board_full(self.board_state)
        if (is_full): # if the board if full of stones and no extra empty spaces, game is finished
            return True
        else:
            return exist
    
    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = ""
        size = len(self.board_state)
        
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]
        numbers = list(range(1, 100))[:size]
        
        label_move = "Move: " + str(self.move) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"
        
        # construct the board output
        out += (label_move + label_letters + label_boundry)
        
        for i in range(size-1,-1,-1):
            line = ""
            line += (str("%2d" % (i+1)) + " |" + " ")
            for j in range(size):
                # check if it's the last move
                line += gomoku_util.color_shape[self.board_state[i][j]]
                if (i,j) == self.last_coord:
                    line += ")"
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out
    
    def encode(self):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        img = np.array(self.board_state) # shape [board_size, board_size]
        return img
