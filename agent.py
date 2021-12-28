import os
import pygame
import pickle
import numpy as np
import scipy.spatial.distance as ds
from tqdm import tqdm

from globalVar import COLOR
from environment import Tileset
from voronoiNbv import VoronoiNBV
import util

# ---- Explored map value definitions ---

class Agent:
    def __init__(self, size, num_tiles, tile_size, sensor_range=7):
        self.pos = None
        self.size = size
        self.num_tiles = num_tiles
        self.tile_size = tile_size
        self.sensor_range = sensor_range
        self.tile = pygame.Surface(self.size)
        self.tile.fill(COLOR[5])
        self.explored_map = np.zeros(num_tiles, dtype=int) + 3
        self.explored_tileset = Tileset(self.tile_size, (0, 1, 2, 3, 4))
        self.sensor_ray_file = "sensorRays/"
        self.getSensorRays()
        self.exp_alg = VoronoiNBV(self.tile_size, sensor_range, self.rays)
        
    def getSensorRays(self):
        # If sensor ray configuration is already computed once, load it from file
        filePath = self.sensor_ray_file + str(self.sensor_range) + ".txt"
        if os.path.isfile(filePath):
            print("Loading sensor rays from file...")
            with open(filePath, "rb") as fp:
                self.rays = pickle.load(fp)
        else:
            print("Calculating sensor rays...")
            self.calculateSensorRays(filePath)
    
    def calculateSensorRays(self, filePath):
        # Calculating the tiles that sensors pass through relative to the agent
        
        ray_paths = []
        sensing_area = np.zeros((self.sensor_range*2+1, self.sensor_range*2+1))
        pos_c = (self.sensor_range*self.tile_size[0] + self.tile_size[0]/2, self.sensor_range*self.tile_size[1] + self.tile_size[1]/2)
        sensing_tiles = np.argwhere(sensing_area==0)
        #for tile in sensing_tiles:
        for i in tqdm(range(len(sensing_tiles))):
            tile = sensing_tiles[i]
            tile_c = (tile[0]*self.tile_size[0] + self.tile_size[0]/2, tile[1]*self.tile_size[1] + self.tile_size[1]/2)
            ray_path = []
            if (pos_c[0] - tile_c[0])**2 + (pos_c[1] - tile_c[1])**2 <= (self.sensor_range*self.tile_size[0])**2:
                for block_tile in sensing_tiles:
                    if util.lineBlockedFull(pos_c, tile_c, [block_tile], self.tile_size):
                        ray_path.append(block_tile.tolist())
            ray_paths.append(ray_path)
        
        # Sort rays by length to agent position
        agent_pos = np.array([[self.sensor_range, self.sensor_range]])
        for i in range(len(ray_paths)):
            ray = np.array(ray_paths[i])
            if ray.size != 0:
                ray = ray[np.argsort(ds.cdist(agent_pos, ray)[0])]
            ray_paths[i] = ray.tolist()
        
        self.rays = ray_paths
        with open(filePath, "wb") as fp:
            pickle.dump(self.rays, fp)
    
    def updatePos(self, pos):
        #print("Update agent position:", pos)
        self.pos = pos
        
    def render(self, screen):
        self.renderAgent(screen)
    
    def renderAgent(self, screen):
        if self.pos != None:
            screen.blit(self.tile, (self.pos[1]*self.size[0], self.pos[0]*self.size[1]))
    
    def renderExploredArea(self, screen):
        m, n = self.explored_map.shape
        for i in range(m):
            for j in range(n):
                tile = self.explored_tileset.tiles[self.explored_map[i,j]]
                screen.blit(tile, (j*self.tile_size[0], i*self.tile_size[1]))
    
    def renderEnvWithExploredArea(self, screen, env):
        visualized_map = np.where(np.logical_and(self.explored_map == 1, env.map == 1), 2, self.explored_map)
        visualized_map = np.where(np.logical_and(env.map == 1, self.explored_map == 3), env.map, visualized_map)
        m, n = visualized_map.shape
        #print(visualized_map)
        for i in range(m):
            for j in range(n):
                tile = self.explored_tileset.tiles[visualized_map[i,j]]
                screen.blit(tile, (j*self.tile_size[0], i*self.tile_size[1]))
            
    def getMovePos(self, mov):
        return (self.pos[0] + mov[0], self.pos[1] + mov[1])
    
    def getSensorData(self, env):
        sensed_tiles = []
        mod = np.array([self.pos[0] - self.sensor_range, self.pos[1] - self.sensor_range])
        print(mod)
        for ray in self.rays:
            for ray_component in ray:
                tile_pos = (ray_component[0]+mod[0], ray_component[1]+mod[1])
                # Make sure the tile is inside the bounds of the env
                if tile_pos[0] < env.num_tiles[0] and tile_pos[1] < env.num_tiles[1]:
                    if tile_pos[0] >= 0 and tile_pos[1] >= 0:
                        sensed_tiles.append(tile_pos)
                        if env.map[tile_pos] == 1:
                            break
        return sensed_tiles
    
    def getSensorData2(self, env):
        return util.getVisibleTiles(self.pos, env.map, self.sensor_range, self.rays)
        
    def gatherData(self, env):
        sensor_data = self.getSensorData2(env)
        for tile_pos in sensor_data:
            self.explored_map[tile_pos] = env.map[tile_pos]
            
    def findFrontiers(self,explored_map):
        return util.findAdjacentValues(explored_map, 3, 0, 4)
            
    def explore(self):
        self.explored_map = self.findFrontiers(self.explored_map)
        path = self.exp_alg.compute(self.explored_map, self.pos)
        if path != None:
            return (path[-2][0] - self.pos[0], path[-2][1] - self.pos[1])
        else:
            return (0,0)
        