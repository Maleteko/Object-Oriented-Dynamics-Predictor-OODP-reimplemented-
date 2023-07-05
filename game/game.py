import random
import pygame
import imp
import numpy as np

import game.monsterkong as mk

class Game:
    def __init__(self, levelNumber):
        map_config = imp.load_source('map_config', 'game/monsterkong/8blocks/level0_' + str(levelNumber) +'.py').map_config

        self.game = mk.MonsterKong(map_config=map_config)

        self.game.screen = pygame.display.set_mode((self.game.height, self.game.width))
        self.game.init()
        self.actions = self.game.getActions()
        self.num_actions = len(self.actions)

    def step(self, action):
        self.game.step(self.actions[action])
        return np.transpose(self.game.getScreenRGB(),(1, 0, 2))

    def randomStep(self):
        rand = random.randrange(self.num_actions)
        action = self.actions[rand]
        self.game.step(action)
        return rand, np.transpose(self.game.getScreenRGB(),(1, 0, 2))