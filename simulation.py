import pygame
from pygame.locals import *
import numpy as np

from environment import Env
from agent import Agent

#---- HOTKEYS SETUP -----
# R: Randomize map
# K: Save map
# L: Load map
# Q: Exit simulation
# P: Place agent
# W: Move agent up
# S: Move agent down
# A: Move agent left
# D: Move agent right

#---- HOTKEYS SIMULATION -----
# K: Save map
# L: Load map
# Q: Exit simulation
# W: Move agent up
# S: Move agent down
# A: Move agent left
# D: Move agent right

STATES = ("setup", "simulation")

class Simulation:
    W = 640
    H = 640
    SIZE = W, H
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(Simulation.SIZE)
        pygame.display.set_caption("NBV2D")
        self.env = Env()
        self.agent = Agent(self.env.tile_size, self.env.num_tiles, self.env.tile_size)
        self.running = True
        self.clock = pygame.time.Clock()
        self.state = STATES[0]
        self.auto_solve = False
        
        self.env.render(self.screen)
        
    def placeAgent(self):
        pos = self.env.getTileIdx(pygame.mouse.get_pos())
        self.agent.updatePos(pos)
        self.moveAgent((0,0))
        print("Placing agent at", pos, "...")
        
    def moveAgent(self, mov):
        pos = self.agent.getMovePos(mov)
        if not self.collision(pos):
            self.agent.updatePos(pos)
            self.agent.gatherData(self.env)
            self.agent.explore()
        
    def collision(self, pos):
        tile_type = self.env.getMapTile(pos)
        if tile_type == 1:
            return True
        else:
            return False
        
    def changeState(self):
        if self.state == STATES[0]:
            self.state = STATES[1]
        elif self.state == STATES[1]:
            self.state = STATES[0]
            
    def toggleAutoSolve(self):
        if self.auto_solve == False:
            print("Auto Solve: ON")
            self.auto_solve = True
        elif self.auto_solve == True:
            print("Auto Solve: OFF")
            self.auto_solve = False
            
    def autoSolve(self):
        #print("Test")
        self.agent.gatherData(self.env)
        mov = self.agent.explore()
        pos = self.agent.getMovePos(mov)
        self.agent.updatePos(pos)
        self.render()
        
    def render(self):
        if self.state == STATES[0]:
            self.env.render(self.screen)
            self.agent.render(self.screen)
        elif self.state == STATES[1]:
            self.env.renderOverlay(self.screen)
            #self.agent.renderExploredArea(self.screen)
            self.agent.renderEnvWithExploredArea(self.screen, self.env)
            self.agent.exp_alg.render(self.screen, self.env.tile_size)
            self.agent.render(self.screen)
        
    def universalStateCommands(self, event):
        if event.type == QUIT:
            self.running = False
            
        elif event.type == KEYDOWN:
            if event.key == K_k:
                self.env.saveMap()
            
            elif event.key == K_l:
                self.env.loadMap()
                
            elif event.key == K_w:
                self.moveAgent((-1,0))
                
            elif event.key == K_s:
                self.moveAgent((1,0))
                
            elif event.key == K_a:
                self.moveAgent((0,-1))
            
            elif event.key == K_d:
                self.moveAgent((0,1))
                
            self.render()
                
            if event.key == K_ESCAPE or event.key == K_q:
                self.running = False
            elif event.key == K_RETURN:
                self.changeState()
            elif event.key == K_SPACE:
                self.toggleAutoSolve()
            
        
    def setupStateCommands(self, event):
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            self.env.makeWall(mouse_pos)
            self.render()
            
        elif pygame.mouse.get_pressed()[2]:
            mouse_pos = pygame.mouse.get_pos()
            self.env.makeFree(mouse_pos)
            self.render()

        elif event.type == KEYDOWN:
            if event.key == K_r:
                self.env.setRandomMap()
            
            elif event.key == K_p:
                self.placeAgent()
                
            self.render()
                
    def simulateStateCommands(self, event):        
        if event.type == KEYDOWN:
            self.render()
            
            
        
    def run(self):
        while self.running:
            
            for event in pygame.event.get():
                self.universalStateCommands(event)
                if self.state == STATES[0]:
                    self.setupStateCommands(event)
                elif self.state == STATES[1]:
                    self.simulateStateCommands(event)
            
            if self.auto_solve:
                self.autoSolve()
                
            pygame.display.flip()
            self.clock.tick(60)
                   
        pygame.quit()
        
        

def main():
    simulation = Simulation()
    simulation.run()
    
main()