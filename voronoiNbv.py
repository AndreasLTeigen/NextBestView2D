import pygame
import numpy as np
import networkx as nx
from skimage import measure
from scipy.spatial import distance
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

import util
from globalVar import COLOR
from environment import Tileset

class VoronoiNBV:
    def __init__(self, tile_size, sensor_r, rays):
        self.tile_size = tile_size
        self.vornoi_tileset = Tileset(self.tile_size, (6, 7))
        self.bfs = BFS()
        self.sensor_r = sensor_r
        self.rays = rays
    
    def compute(self, explored_map, agent_pos):
        self.labelFrontiers(explored_map)
        self.findFrontierEdges(self.frontier_label_map)
        #self.getFrontierVoronoi(self.frontier_edges)
        self.getFrontierVoronoi2(explored_map, self.frontier_edges, self.frontier_map)
        #self.getFrontierVoronoiNew(self.frontier_edges, explored_map)
        #self.voronoiMap(explored_map)
        return self.getShortestPathToClosestFrontier(explored_map, agent_pos)
        
    
    def labelFrontiers(self, explored_map):
        self.frontier_map = np.where(explored_map == 4, 1, 0)                       # Binary frontier map
        self.frontier_label_map = measure.label(self.frontier_map, connectivity=2)  # Distinguish frontiers
        
    def findFrontierEdges(self, frontier_label_map):
        # Find the two outerpoints of all frontiers, assuming they are labeled
        self.frontier_edges = []
        (unique, counts) = np.unique(frontier_label_map, return_counts=True)
        unique = np.delete(unique,0) # Deleting the 0 value from the unique array as that is no frontier
        for frontier_nr in unique:
            front_coords = np.where(frontier_label_map == frontier_nr)
            front_coords = list(zip(front_coords[0], front_coords[1]))
            longest_dist = 0
            tile_pairs = []
            if len(front_coords) == 1:
                tile_pairs = [front_coords[0], front_coords[0]]
            else:
                for tile1 in front_coords:
                    for tile2 in front_coords:
                        dist = distance.euclidean(tile1, tile2)
                        if dist > longest_dist:
                            longest_dist = dist
                            tile_pairs = [tile1, tile2]

            self.frontier_edges.append(tile_pairs)
        return self.frontier_edges
            
    def getFrontierVoronoi(self, frontier_edges):
        self.frontier_voronoi = []
        for tile_pairs in frontier_edges:
            midpoint = util.getLineMidpoint(tile_pairs[0], tile_pairs[1])
            self.frontier_voronoi.append(midpoint)
        return self.frontier_voronoi
    
    def getFrontierVoronoi2(self, explored_map, frontier_edges, frontier_map):
        m,n = explored_map.shape
        self.frontier_voronoi = []
        frontier_visibility_map = self.getFrontierVisibilityMap(explored_map, frontier_map)
        #print(frontier_visibility_map)
        for tile_pairs in frontier_edges:
            dist = distance.euclidean(tile_pairs[0], tile_pairs[1])
            v1,v2,h1,h2 = util.getIndices(tile_pairs[0], tile_pairs[1])
            if dist < (2*self.sensor_r - 2):
                search_area = frontier_visibility_map[v1:v2, h1:h2]
                flat_index = np.argmax(search_area)
                voronoi = util.flattenedIndex2Touple(flat_index, search_area.shape)
                voronoi = (voronoi[0]+v1, voronoi[1]+h1)
                self.frontier_voronoi.append(voronoi)
            else:
                for tile in tile_pairs:
                    v1, v2 = max(0, tile[0]-self.sensor_r + 1), min(m, tile[0]+self.sensor_r)
                    h1, h2 = max(0, tile[1]-self.sensor_r + 1), min(n, tile[1]+self.sensor_r)
                    search_area = frontier_visibility_map[v1:v2, h1:h2]
                    #print(search_area)
                    flat_index = np.argmax(search_area)
                    voronoi = util.flattenedIndex2Touple(flat_index, search_area.shape)
                    voronoi = (voronoi[0]+v1, voronoi[1]+h1)
                    self.frontier_voronoi.append(voronoi)
                
    def getFrontierVisibilityMap(self, explored_map, frontier_map):
        all_frontier_visibility_map = np.zeros(explored_map.shape)
        #obstacle_map = np.where(np.logical_or(explored_map == 1, explored_map == 3), 1, 0)
        obstacle_map = np.where(explored_map == 1, 1, 0)
        #front_coords = np.where(explored_map == 4)
        front_coords = np.where(util.findAdjacentValues(explored_map, 4, 3, 9) == 9)
        front_coords = list(zip(front_coords[0], front_coords[1]))
        for front_coord in front_coords:
            visible_tiles = util.getVisibleTiles(front_coord, obstacle_map, self.sensor_r, self.rays)
            frontier_visibility_map = np.zeros(explored_map.shape)
            for tile in visible_tiles:
                if frontier_visibility_map[tile] == 0:
                    frontier_visibility_map[tile] = 1
            all_frontier_visibility_map = np.where(frontier_visibility_map == 1, all_frontier_visibility_map + 1, all_frontier_visibility_map)
        #all_frontier_visibility_map = np.where(obstacle_map == 0, all_frontier_visibility_map, 0)
        all_frontier_visibility_map = np.where(np.logical_or(obstacle_map == 0, explored_map == 3), all_frontier_visibility_map, 0)
        return all_frontier_visibility_map
    
    def getFrontierVoronoiNew(self, frontier_edges, explored_map):
        self.frontier_voronoi = []
        for tile_pairs in frontier_edges:
            dist = distance.euclidean(tile_pairs[0], tile_pairs[1])
            midpoint = util.getLineMidpoint(tile_pairs[0], tile_pairs[1])
            frontier_voronoi_tile = explored_map[(int(midpoint[0]), int(midpoint[1]))]
            if dist < (2*self.sensor_r) and ( frontier_voronoi_tile == 0 or frontier_voronoi_tile == 4):
                self.frontier_voronoi.append(midpoint)
            else:
                #print("HERHEHR")
                #print("Dist: ", dist)
                #print("Midpoint: ", (midpoint[0], midpoint[1]))
                self.findCardinalFrontierVoronoi(tile_pairs, explored_map)
        
        return self.frontier_voronoi
    
    def findCardinalFrontierVoronoi(self, tile_pair, explored_map):
        m,n = explored_map.shape
        tile1 = tile_pair[0]
        tile2 = tile_pair[1]
        x_y_dist = (tile1[0] - tile2[0], tile1[1] - tile2[1])
        
        #First looking horizontally
        print(np.sign(x_y_dist[1]))
        print(np.sign(x_y_dist[0]))
        if np.sign(x_y_dist[1]) == -1:
            midpoint_h1 = (tile1[0], int((min(tile1[1]+self.sensor_r-1, n) - tile1[0])/2))
            midpoint_h2 = (tile2[0], int((tile2[1] - max(tile2[1]+self.sensor_r-1, 0))/2))
        elif np.sign(x_y_dist[1]) == 1:
            midpoint_h1 = (tile1[0], int((tile1[1] - max(tile1[1]+self.sensor_r-1, 0))/2))
            midpoint_h2 = (tile2[0], int((min(tile2[1]+self.sensor_r-1, n) - tile2[0])/2))
        
        if np.sign(x_y_dist[0]) == -1:
            midpoint_v1 = (int((min(tile1[0]+self.sensor_r-1, m)-tile1[0])/2), tile1[1])
            midpoint_v2 = (int((tile2[0] - max(tile2[0]+self.sensor_r-1, 0))/2), tile2[1])
        
        elif np.sign(x_y_dist[0]) == 1:
            midpoint_v1 = (int((tile1[0] - max(tile1[0]+self.sensor_r-1, 0))/2), tile1[1])
            midpoint_v2 = (int((min(tile2[0]+self.sensor_r-1, m)-tile2[0])/2), tile2[1])
        
        self.frontier_voronoi.append(midpoint_h1)
        self.frontier_voronoi.append(midpoint_v1)
        self.frontier_voronoi.append(midpoint_h2)
        self.frontier_voronoi.append(midpoint_v2)
        print(midpoint_h1)
        print(midpoint_v1)
        print(midpoint_h2)
        print(midpoint_v2)
        #print("Tiles: ", tile_pair)
        #print("Dist x,y: ", x_y_dist)
        
        
    def findFrontierEdgesNPHARD(self):
        # Find the two outerpoints of all frontiers, assuming they are labeled
        # NOTE: A tile can have 3 or more outer points
        num_tiles = self.frontier_label_map.shape
        self.frontier_edges = []
        (unique, counts) = np.unique(self.frontier_label_map, return_counts=True)
        unique = np.delete(unique,0) # Deleting the 0 value from the unique array as that is no frontier
        counts = np.delete(counts,0)
        for frontier_nr in unique:
            front_coords = np.where(self.frontier_label_map == frontier_nr)
            front_coords = list(zip(front_coords[0], front_coords[1]))
            tile_pairs = []
            idx = np.where(unique == frontier_nr)
            if counts[idx] == 1:
                tile_pairs.append(front_coords[0])
                tile_pairs.append(front_coords[0])
            else:
                for tile in front_coords:
                    i_min, i_max = max(0,tile[0]-1), min(num_tiles[0], tile[0]+2)
                    j_min, j_max = max(0,tile[1]-1), min(num_tiles[1], tile[1]+2)
                    (unique_n, counts_n) = np.unique(self.frontier_label_map[i_min:i_max, j_min:j_max], return_counts=True)
                    idx_n = np.where(unique_n == frontier_nr)
                    if counts_n[idx_n] == 2:
                        tile_pairs.append(tile)
                self.frontier_edges.append(tile_pairs)
             
       
    def getShortestPathToClosestFrontier(self, explored_map, agent_pos):
        obstruction_map = np.where(explored_map == 0, 0, 1)
        obstruction_map = np.where(np.logical_or(explored_map == 3, explored_map == 4), 0, obstruction_map)
        path = self.bfs.solve(obstruction_map, agent_pos, self.frontier_voronoi)
        return path
                
    def voronoiMap(self, explored_map):
        m, n = explored_map.shape
        points = np.where(explored_map == 1)
        points = list(zip(points[0]+0.5, points[1]+0.5)) 
        vor = Voronoi(points)
        voronoi_points = vor.vertices
        self.voronoi_vertices = []
        for vertice in voronoi_points:
            i_min, i_max = max(0, int(vertice[0])-1), min(m, int(vertice[0])+2)
            j_min, j_max = max(0, int(vertice[1])-1), min(n, int(vertice[1])+2)
            neighbour_walls = np.where(explored_map[i_min:i_max, j_min:j_max] == 1)
            neighbour_walls = list(zip(neighbour_walls[0], neighbour_walls[1])) 
            if neighbour_walls:
                add_vertex = True
                for wall in neighbour_walls:
                    dist = distance.euclidean(vertice, (wall[0]+i_min+0.5, wall[1]+j_min+0.5))
                    if dist < 0.75:
                        add_vertex = False
                        break
                if add_vertex:
                    self.voronoi_vertices.append(vertice)
            else:
                self.voronoi_vertices.append(vertice)
                
        #fig = voronoi_plot_2d(vor)
        #plt.show()
        
    #def explorationScenario1(self, explored_map, frontier_map, frontier_label_map):
    #    a = np.where(np.logical_)
    #    frontier_walls = util.findAdjacentValues(a, v1, v2, marker_value)
    
    def render(self, screen, tile_size):
        self.visualizeFrontierEdges(screen, tile_size)
        self.visualizeFrontierVoronoi(screen, tile_size)
        #self.visualizeVoronoiPoints(screen, tile_size)
    
    def visualizeFrontierEdges(self, screen, tile_size):
        for tile_pair in self.frontier_edges:
            tile_c1 = (int(tile_pair[0][0]*tile_size[0] + tile_size[0]/2), int(tile_pair[0][1]*tile_size[1] + tile_size[1]/2))
            tile_c2 = (int(tile_pair[1][0]*tile_size[0] + tile_size[0]/2), int(tile_pair[1][1]*tile_size[1] + tile_size[1]/2))
            pygame.draw.circle(screen, COLOR[6], (tile_c1[1], tile_c1[0]), int(tile_size[1]/2))
            pygame.draw.circle(screen, COLOR[6], (tile_c2[1], tile_c2[0]), int(tile_size[1]/2))
            
    def visualizeFrontierVoronoi(self, screen, tile_size):
        for point in self.frontier_voronoi:
            tile_coord = (int(point[0])*tile_size[0], int(point[1])*tile_size[1])
            #screen.blit(self.vornoi_tileset.tiles[1], (tile_coord[1], tile_coord[0]))
            acc_coord = (int(int(point[0])*tile_size[0] + tile_size[0]/2), int(int(point[1])*tile_size[1] + tile_size[1]/2))
            pygame.draw.circle(screen, COLOR[7], (acc_coord[1], acc_coord[0]), int(tile_size[1]/4))
    
    def visualizeVoronoiPoints(self, screen, tile_size):
        for point in self.voronoi_vertices:
            acc_coord = (int(point[0]*tile_size[0]), int(point[1]*tile_size[1]))
            pygame.draw.circle(screen, COLOR[7], (acc_coord[1], acc_coord[0]), int(tile_size[1]/4))
            tile_coord = (int(point[0])*tile_size[0], int(point[1])*tile_size[1])
            screen.blit(self.vornoi_tileset.tiles[1], (tile_coord[1], tile_coord[0]))
        
        

class BFS:
    def solve(self, a, start, goals):
        # Todo: Freezes when goal is overlapping with obstacle 
        goals = util.make2DListInt(goals)
        start_l = [start[0], start[1]]
        #print("Start: ", start)
        #print("Start_l: ", start_l)
        #print("Goal: ", goals)
        goals_x = []
        goals_y = []
        
        if start_l in goals:
            goals.remove(start_l)
            
        if not goals:
            return None
                
        for goal in goals:
            goals_x.append(int(goal[1]))
            goals_y.append(int(goal[0]))
        
        steps = np.zeros((a.shape[0], a.shape[1]))
        steps[start] = 1
        k = 1
        #print(steps)
        while not np.any(steps[goals_y, goals_x]):
            steps = util.findAdjacentValues(steps, k, 0, k+1)
            steps = np.where(a == 1, 0, steps)
            #print(k)
            ##print(goals)
            #print(a)
            #print(steps)
            k += 1
            #print(steps[goals_y, goals_x])
            #print(not np.any(steps[goals_y, goals_x]))
            
        
        for goal in goals:
            if steps[(int(goal[0]), int(goal[1]))] != 0:
                end_point = (int(goal[0]), int(goal[1]))
                break
        #print(end_point)
        m, n = a.shape
        path = []
        temp_pos = end_point
        path.append(temp_pos)
        k -= 1
        #print(temp_pos)
        while k > 0:
            i_min, i_max = max(0, temp_pos[0]-1), min(temp_pos[0]+2, m)
            j_min, j_max = max(0, temp_pos[1]-1), min(temp_pos[1]+2, n)
            neighbourhood = steps[i_min:i_max, j_min:j_max]
            #print(neighbourhood)
            temp_pos = np.where(neighbourhood == k)
            temp_pos = list(zip(temp_pos[0], temp_pos[1]))[0]
            #print(i_min, j_min)
            temp_pos = (temp_pos[0]+i_min, temp_pos[1]+j_min)
            #print("Final temp pos: ", temp_pos)
            path.append(temp_pos)
            k -= 1
        #print(path)
        return path
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
'''
    def findFrontiers(self, explored_map):
        # Frontiers is where free spaces meets unexplored space
        temp_map = np.where(np.logical_or(explored_map == 0, explored_map == 3), explored_map, -1)
        diff_map1 = np.diff(temp_map, axis=0)
        diff_map2 = np.diff(temp_map, axis=1)
        
        frontier_map1 = np.where(util.appendZeroRow(diff_map1) == -3, 1, 0)
        frontier_map2 = np.where(util.appendZeroColumn(diff_map2) == -3, 1, 0)
        frontier_map3 = np.where(util.preappendZeroRow(diff_map1) == 3, 1, 0)
        frontier_map4 = np.where(util.preappendZeroColumn(diff_map2) == 3, 1, 0)
        
        frontier_map1 = np.bitwise_or(frontier_map1, frontier_map2)
        frontier_map2 = np.bitwise_or(frontier_map3, frontier_map4)
        frontier_map = np.bitwise_or(frontier_map1, frontier_map2)
        explored_map = np.where(frontier_map == 1, 4, explored_map)
        return explored_map
'''