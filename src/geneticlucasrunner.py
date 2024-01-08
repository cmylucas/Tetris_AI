from copy import copy, deepcopy
import numpy as np
import pygame
from piece import BODIES, Piece
from board import Board
import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
from collections import deque
from network import QNetwork
from statistics import mean, median
from tqdm import tqdm
from lucas_helpers import *
# from logs import *
from datetime import datetime
from time import sleep
import pickle
from geneticlucas import Lucas_AI
from game import Game

def run_generations():
    generations = 2000
    population_size = 20
    population = []
    log_every = 1

    for i in range(population_size):
        bot = Lucas_AI(genotype = None, state_size = 4)
        population.append(bot)

    for generation in tqdm(range(generations)):
        # print(f"Generation {generation}")
        
        total_fitness = 0
        for bot in tqdm(population):
            game = Game("geneticlucas", agent = bot)
            piece_dropped, rows_cleared = game.run_no_visual()
            bot.fitness = piece_dropped + (rows_cleared)*10
            total_fitness += bot.fitness
            bot.lines_cleared = rows_cleared

        for bot in population:
            bot.relative_fitness = bot.fitness/total_fitness
            
        next_population = []
        population = sorted(population, key=lambda x: x.fitness, reverse=True)

        # top 5 elite bots
        for x in range(5):
            next_population.append(population[x])

        weights = [bot.relative_fitness for bot in population]

        # weighted random choice of parents
        for x in range(5,population_size):
            parent1, parent2 = random.choices(population, weights, k=2)
            child = parent1.breed(parent2, mutation_rate = 0.1)
            next_population.append(child)

        with open('geneticmodel.pkl', 'wb') as f:
            pickle.dump(population[0], f)

        if generation % log_every == 0:
            with open('log.txt', 'a') as f:
                f.write(f'generation: {generation}, best fitness: {population[0].fitness}, {population[0].lines_cleared}, best genotype: {population[0].genotype}, average fitness: {total_fitness/100}\n')

        population = next_population
    

if __name__ == "__main__":
    run_generations()
