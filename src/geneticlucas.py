from copy import copy, deepcopy
import numpy as np
import pygame
from piece import BODIES, Piece
from board import Board
import random
from collections import deque
from network import QNetwork
from statistics import mean, median
from tqdm import tqdm
# from genetic_helpers import *
from keras.models import load_model
# from lucastrainer import *
from lucas_helpers import *


class Lucas_AI:
    def __init__(self, genotype = None, state_size = 4):
        # genotype -> weights
        # fitness -> score/rows cleared
        # breed -> cross and mutate
        # [lines_cleared, bumpiness, total_height, number_holes]
        self.state_size = state_size
        self.genotype = np.array(genotype)
        if(genotype is None):
            self.genotype = np.array([random.uniform(-1, 1) for _ in range(self.state_size)])
        # if genotype is None:
        #     self.genotype = np.array([0.76, -0.18, -0.5, -0.35])
        self.fitness = 0
        self.lines_cleared = 0
        # ideal weights = [0.76, -0.18, -0.5, -0.35]
        # self.mutation_rate = mutation_rate

    def get_state(self, board, piece, x, y):
        lines = board.clear_rows()
        board_copy = deepcopy(board.board)
        
        for pos in piece.body:
            board_copy[y + pos[1]][x + pos[0]] = True
        
        np_board = bool_to_np(board_copy)
        # print(np_board)
        peaks = get_peaks(np_board)
        bumpiness = get_bumpiness(peaks)
        holes = get_holes(peaks, np_board)
        total_height = 0
        for i in range(10):
            total_height += peaks[i]
        # print(f"peaks: {peaks}")
        total_holes = 0
        for i in range(10):
            total_holes += holes[i]
        # print(f"holes: {holes}")
        # if lines!=0:
        #     print(lines)
        return [lines, bumpiness, total_height, float(total_holes)]


    def breed(self, agent, mutation_rate):
        genotype = [0,0,0,0]
        for i in range(self.state_size):
            genotype[i] = self.genotype[i] if random.getrandbits(1) else agent.genotype[i]

        for i in range(self.state_size):
            if random.random() < mutation_rate:
                genotype[i] = random.random()
        child = Lucas_AI(genotype, 4)
        return child
        
    def get_best_move(self, board, piece):
        best_action = []
        best_value = -999999999
        for rotation in range(4):
            piece = piece.get_next_rotation()
            for x in range(board.width):
                try:
                    y = board.drop_height(piece, x)
                except:
                    continue
                newboard = deepcopy(board)
                newboard.place(x,y,piece)
                newpiece = deepcopy(piece)
                state = self.get_state(newboard, newpiece, x, y)
                # print(f"genetic states: {state}")
                # print(f"genetic loop: {x, y}, {piece}")
                state_value = 0
                for i in range(self.state_size):
                    state_value += (state[i] * self.genotype[i])
                
                if state_value >= best_value:
                    best_value = state_value
                    newpiece = deepcopy(piece)
                    best_action = [x, newpiece]

        return best_action[0], best_action[1]
        
    def __str__(self):
        return f"fitness = {self.fitness}, genotype = {self.genotype}"





