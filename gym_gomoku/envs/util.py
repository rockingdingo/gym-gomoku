import numpy as np
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO
import sys
import six

class GomokuUtil(object):
    
    def __init__(self):
        # default setting
        self.BLACK = 'black'
        self.WHITE = 'white'
        self.color = [self.BLACK, self.WHITE]
        self.color_dict = {'empty': 0, 'black': 1, 'white': 2}
        self.color_dict_rev = {v: k for k, v in self.color_dict.items()}
        self.color_shape = {0: '.', 1: 'X', 2: 'O'}
    
    def other_color(self, color):
        '''Return the opositive color of the current player's color
        '''
        assert color in self.color, 'Invalid player color'
        opposite_color = self.color[0] if color == self.color[1] else self.color[1]
        return opposite_color
    
    def iterator(self, board_state):
        ''' Iterator for 2D list board_state
            Return: Row, Column, diagnoal, list of coordinate tuples, [(x1, y1), (x2, y2), ...,()], (6n-2-16) lines
        '''
        list = []
        size = len(board_state)
        
        # row
        for i in range(size): # [(i,0), (i,1), ..., (i,n-1)]
            list.append([(i, j) for j in range(size)])
        
        # column
        for j in range(size):
            list.append([(i, j) for i in range(size)])
        
        # diagonal: left triangle
        for k in range(size):
            # lower_line consist items [k][0], [k-1][1],...,[0][k]
            # upper_line consist items [size-1][size-1-k], [size-1-1][size-1-k +1],...,[size-1-k][size-1]
            lower_line = [((k-k1), k1) for k1 in range(k+1)]
            upper_line = [((size-1-k2), (size-k-1+k2)) for k2 in range(k+1)]
            if (k == (size-1)): # one diagnoal, lower_line same as upper_line
                list.append(lower_line)
            else :
                if (len(lower_line)>=5):
                    list.append(lower_line)
                if (len(upper_line)>=5):
                    list.append(upper_line)
        
        # diagonal: right triangle
        for k in range(size):
            # lower_line consist items [0][k], [1][k+1],...,[size-1-k][size-1]
            # upper_line consist items [k][0], [k+1][1],...,[size-1][size-1-k]
            lower_line = [(k1, k + k1) for k1 in range(size-k)]
            upper_line = [(k + k2, k2) for k2 in range(size-k)]
            if (k == 0): # one diagnoal, lower_line same as upper_line
                list.append(lower_line)
            else :
                if (len(lower_line)>=5):
                    list.append(lower_line)
                if (len(upper_line)>=5):
                    list.append(upper_line)
        
        for line in list:
            yield line
    
    def value(self, board_state, coord_list):
        ''' Fetch Value from 2D list with coord_list
        '''
        val = []
        for (i,j) in coord_list:
            val.append(board_state[i][j])
        return val
    
    def check_five_in_row(self, board_state):
        ''' Args: board_state 2D list
            Return: exist, color
        '''
        size = len(board_state)
        black_pattern = [self.color_dict[self.BLACK] for _ in range(5)] # [1,1,1,1,1]
        white_pattern = [self.color_dict[self.WHITE] for _ in range(5)] # [2,2,2,2,2]
        
        exist_final = False
        color_final = "empty"
        black_win, _ = self.check_pattern(board_state, black_pattern)
        white_win, _ = self.check_pattern(board_state, white_pattern)
        
        if (black_win and white_win):
            raise error.Error('Both Black and White has 5-in-row, rules conflicts')
        # Check if there is any one party wins
        if not (black_win or white_win):
            return exist_final, "empty"
        else:
            exist_final = True
        if (black_win):
            return exist_final, self.BLACK
        if (white_win):
            return exist_final, self.WHITE
    
    def check_board_full(self, board_state):
        is_full = True
        size = len(board_state)
        for i in range(size):
            for j in range(size):
                if (board_state[i][j]==0):
                    is_full = False
                    break
        return is_full
    
    def check_pattern(self, board_state, pattern):
        ''' Check if pattern exist in the board_state lines,
            Return: exist: boolean
                    line: coordinates that contains the patterns
        '''
        exist = False
        pattern_found = [] # there maybe multiple patterns found
        for coord in self.iterator(board_state):
            line_value = self.value(board_state, coord)
            if (self.is_sublist(line_value, pattern)):
                exist = True
                pattern_found.append(coord)
        return exist, pattern_found
    
    def check_pattern_index(self, board_state, pattern):
        '''Return the line contains the pattern, and its start index of the pattern
        '''
        start = -1
        startlist = []
        exist_patttern, lines = self.check_pattern(board_state, pattern)
        if (exist_patttern):
            for line in lines:
                start = self.index(self.value(board_state, line), pattern)
                startlist.append(start)
            return lines, startlist  # line: list[list[(x1, y1),...]], startlist: list[int]
        else: # pattern not found
            return None, startlist
    
    def is_sublist(self, list, sublist):
        l1 = len(list)
        l2 = len(sublist)
        is_sub = False
        for i in range(l1):
            curSub = list[i: min(i+l2, l1)]
            if (curSub == sublist): # check list equal
                is_sub = True
                break
        return is_sub
    
    def index(self, list, sublist):
        ''' Return the starting index of the sublist in the list
        '''
        idx = - 1
        l1 = len(list)
        l2 = len(sublist)
        
        for i in range(l1):
            curSub = list[i: min(i+l2, l1)]
            if (curSub == sublist): # check list equal
                idx = i
                break
        return idx

gomoku_util = GomokuUtil()
# Rule.other_color('black')

### Opponent policies ###
def make_random_policy(np_random):
    ''' Get the random action ID of all the empty legal moves, prev_state and prev_action not used
    '''
    def random_policy(curr_state, prev_state, prev_action):
        b = curr_state.board
        legal_moves = b.get_legal_move()
        next_move = legal_moves[np_random.choice(len(legal_moves))]
        return b.coord_to_action(next_move[0], next_move[1])
    return random_policy

def make_beginner_policy(np_random):
    '''General Rules for playing gomoku
    '''
    
    def defend_policy(curr_state):
        '''Return the action Id, if defend situation is needed
        '''
        b = curr_state.board
        player_color = curr_state.color
        opponent_color = gomoku_util.other_color(player_color)
        lines, start, next_move = None, None, None # initialization
        
        ## Defend Strategy, check if these patterns exist:
        # Four in a Row A: [empty,color,color,color,color]
        pattern_four_a = [0] + [gomoku_util.color_dict[opponent_color]] * 4 
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_four_a)
        if (len(start)>= 1): # At least 1 found
            next_move = lines[0][start[0]]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        
        # Four in a Row B: [color,color,color,color,empty]
        pattern_four_b = [gomoku_util.color_dict[opponent_color]] * 4 + [0]
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_four_b)
        if (len(start)>= 1): # At least 1 found
            next_move = lines[0][start[0] + 4]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        
        # Three in a Row: [empty,color,color,color,empty] 
        pattern_three = [0] + [gomoku_util.color_dict[opponent_color]] * 3 + [0]
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_three)
        if (len(start) >= 1):
            line = lines[0] # deal with the first three-in-a-row
            start_idx = start[0]
            moves = [line[start_idx], line[start_idx + 4]]
            next_move = moves[np_random.choice(len(moves))]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        return None
    
    def strike_policy_B(curr_state, prev_state, prev_action):
        b = curr_state.board
        all_legal_moves = b.get_legal_move()
        
        # last action taken by the oppenent
        last_action = prev_state.board.last_action 
        last_coord = prev_state.board.last_coord
        player_color = curr_state.color
        
        # 4 player_color in a row, [empty,color,color,color,color]
        pattern_four_a = [0] + [gomoku_util.color_dict[player_color]] * 4 
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_four_a)
        if (len(start)>= 1): # At least 1 found
            next_move = lines[0][start[0]]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        
        # 4 in a Row B: [color,color,color,color,empty]
        pattern_four_b = [gomoku_util.color_dict[player_color]] * 4 + [0]
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_four_b)
        if (len(start)>= 1): # At least 1 found
            next_move = lines[0][start[0] + 4]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        
        # 3 in a row, place one, add to 4
        pattern_three = [0] + [gomoku_util.color_dict[player_color]] * 3 + [0]
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_three)
        if (len(start) >= 1):
            line = lines[0] # deal with the first three-in-a-row
            start_idx = start[0]
            moves = [line[start_idx], line[start_idx + 4]]
            next_move = moves[np_random.choice(len(moves))]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        
        # 2 in a row, empty place one, add to 3
        pattern_two = [0] + [gomoku_util.color_dict[player_color]] * 2 + [0]
        lines, start = gomoku_util.check_pattern_index(b.board_state, pattern_two)
        if (len(start) >= 1):
            line = lines[0] # deal with the first three-in-a-row
            start_idx = start[0]
            moves = [line[start_idx], line[start_idx + 3]]
            next_move = moves[np_random.choice(len(moves))]
            return b.coord_to_action(next_move[0], next_move[1])
        else:
            lines, start, next_move = None, None, None
        
        # no other coord found, place around the box within previous move
        if (last_coord[0] >=0): # not the first move of 'white' (-1,-1)
            box = [(i,j) for i in range(last_coord[0]-1, last_coord[0]+ 2) for j in range(last_coord[1]-1, last_coord[1] + 2)] # 3x3 box
            legal_moves = []
            for coord in box:
                if (coord in all_legal_moves):
                    legal_moves.append(coord)
            if (len(legal_moves) ==0):
                return None
            else :
                next_move = legal_moves[np_random.choice(len(legal_moves))]
                return b.coord_to_action(next_move[0], next_move[1])
        else:
            legal_moves = b.get_legal_move() # choose from all legal moves
            next_move = legal_moves[np_random.choice(len(legal_moves))]
            return b.coord_to_action(next_move[0], next_move[1])
    
    def strike_policy_A(curr_state, prev_action):
        '''Return the random action Id within the box of prev_action
        '''
        b = curr_state.board
        all_legal_moves = b.get_legal_move()
        
        if prev_action is None:
            # initial game on empty board, random select a position
            next_move = all_legal_moves[np_random.choice(len(all_legal_moves))]
            return b.coord_to_action(next_move[0], next_move[1])
        
        # place a stone around previous move
        prev_coord = b.action_to_coord(prev_action)
        box = [(i,j) for i in range(prev_coord[0]-1, prev_coord[0]+ 2) for j in range(prev_coord[1]-1, prev_coord[1] + 2)] # 3x3 box
        legal_moves = []
        for coord in box:
            if (coord in all_legal_moves):
                legal_moves.append(coord)
        
        # If there is no legal moves around the prev_action position
        if (len(legal_moves) ==0):
            return None
        else :
            next_move = legal_moves[np_random.choice(len(legal_moves))]
            return b.coord_to_action(next_move[0], next_move[1])
    
    def beginner_policy(curr_state, prev_state, prev_action):
        b = curr_state.board
        player_color = curr_state.color
        opponent_color = gomoku_util.other_color(player_color)
        next_move = None # initialization, (x1, y1)
        
        # If defend needed
        action_defend = defend_policy(curr_state)
        if action_defend is not None:
            return action_defend
        
        # No Defend Strategy Met, Use Strike policy B to connect a line
        action_strike = strike_policy_B(curr_state, prev_state, prev_action)
        if action_strike is not None:
            return action_strike
        
        # random choose legal actions
        legal_moves = b.get_legal_move()
        next_move = legal_moves[np_random.choice(len(legal_moves))]
        return b.coord_to_action(next_move[0], next_move[1])
    
    return beginner_policy

def make_medium_policy():
    def medium_policy():
        '''To Do'''
        return 0
    return medium_policy

def make_expert_policy():
    def expert_policy():
        '''To Do'''
        return 0
    return expert_policy

