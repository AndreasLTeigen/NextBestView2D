import pygame
import numpy as np

import util
from globalVar import COLOR

class Tileset:
    def __init__(self, size, tile_selection=None):
        self.size = size
        self.tile_selection = tile_selection
        if self.tile_selection == None:
            self.tile_selection = (0,1,2,3,4,5,6)
        self.generateTileset()
            
    def generateTileset(self):
        self.tiles = []
        for i in self.tile_selection:
            tile = pygame.Surface(self.size)
            tile.fill(COLOR[i])
            self.tiles.append(tile)
            



class Env:
    def __init__(self, num_tiles=(32,32), window_size=(640,640)):
        self.window_size = window_size
        self.num_tiles = num_tiles
        self.tile_size = (window_size[0]/num_tiles[0], window_size[1]/num_tiles[1])
        self.map_tileset = Tileset(self.tile_size, (0,1))
        self.map = np.zeros(num_tiles, dtype=int)
        self.save_folder = "maps/"
        
    def render(self, screen):
        m, n = self.map.shape
        for i in range(m):
            for j in range(n):
                tile = self.map_tileset.tiles[self.map[i,j]]
                screen.blit(tile, (j*self.tile_size[0], i*self.tile_size[1]))
                
    def renderOverlay(self, screen):
        m, n = self.map.shape
        for i in range(m):
            for j in range(n):
                if self.map[i,j] == 1:
                    tile = self.map_tileset.tiles[self.map[i,j]]
                    screen.blit(tile, (j*self.tile_size[0], i*self.tile_size[1]))
                
    def setRandomMap(self):
        n = len(self.map_tileset.tiles)
        self.map = np.random.randint(n, size=self.num_tiles)
        print(self.map)
        
    def getTileIdx(self, px_pos):
        i = px_pos[1] // self.tile_size[1]
        j = px_pos[0] // self.tile_size[0]
        return (int(i), int(j))
    
    def getMapTile(self, pos):
        return self.map[pos]
    
    def changeTile(self, idx_pos, value):
        self.map[idx_pos] = value
        
    def getInSquareWallTiles(self, pos, side):
        u = max(pos[0] - side, 0)
        l = max(pos[1] - side, 0)
        d = min(pos[0] + side, self.num_tiles[0]*self.tile_size[0])
        r = min(pos[1] + side, self.num_tiles[1]*self.tile_size[1])
        
        wall_tiles = np.argwhere(self.map[u:d,l:r] == 1)
        wall_tiles += [u,l]
        
        return wall_tiles
    
    def lineBlocked(self, pt1, pt2, wall_tiles):
        # True / False
        for tile in wall_tiles:
            wall_lines = util.getSquareLines(tile, self.tile_size)
            wall_lines = util.getFurthest2Lines(pt1, wall_lines)
            for line in wall_lines:
                if util.intersect((pt1,pt2), line):
                    return True
        return False
        
    def makeFree(self, px_pos):
        idx_pos = self.getTileIdx(px_pos)
        self.changeTile(idx_pos, 0)
    
    def makeWall(self, px_pos):
        idx_pos = self.getTileIdx(px_pos)
        self.changeTile(idx_pos, 1)
    
    def saveMap(self):
        filename = input("Save map name: ")
        filepath = self.save_folder + filename
        np.save(filepath, self.map)
        print("Map saved!")
        
    def loadMap(self):
        filename = input("Load map name: ")
        filepath = self.save_folder + filename + ".npy"
        self.map = np.load(filepath)
        print("Map loaded!")