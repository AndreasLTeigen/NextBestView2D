import numpy as np

def getLine(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    a = (y1-y2)/(x1-x2)
    b = (x1*y2 - x2*y1)/(x1-x2)
    return (a,b)

def getLineMidpoint(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

def getSquareLines(tile_pos, tile_size):
    #           l1
    #       c1 --- c2
    #    l2 |       | l3
    #       c3 --- c4
    #           l4
    
    c1 = (tile_pos[0]*tile_size[0], tile_pos[1]*tile_size[1])
    c2 = (tile_pos[0]*tile_size[0], (tile_pos[1]+1)*tile_size[1])
    c3 = ((tile_pos[0]+1)*tile_size[0], tile_pos[1]*tile_size[1])
    c4 = ((tile_pos[0]+1)*tile_size[0], (tile_pos[1]+1)*tile_size[1])
    
    return ((c1,c2), (c1,c3), (c2,c4), (c3,c4))

def getFurthest2Lines(pos, lines):
    A, B, C, D = lines
    closest_point = None
    closest_point_dist = np.inf
    # Find the closest point
    for line in lines:
        for point in line:
            point_dist = np.linalg.norm(np.array(pos) - np.array(point))
            if point_dist < closest_point_dist:
                closest_point = point
                closest_point_dist = point_dist
    # Remove lines associated with closest point
    ret_lines = []
    for line in lines:
        if line[0] == closest_point or line[1] == closest_point:
            continue
        ret_lines.append(line)
    return ret_lines
        

def ccw(A,B,C):
    A_y, A_x = A[0], A[1]
    B_y, B_x = B[0], B[1]
    C_y, C_x = C[0], C[1]
    return (C_y-A_y) * (B_x-A_x) > (B_y-A_y) * (C_x-A_x)

# Return true if line segments AB and CD intersect
def intersect(line1,line2):
    A, B = line1[0], line1[1]
    C, D = line2[0], line2[1]
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def lineBlockedFull(pt1, pt2, wall_tiles, tile_size):
        # True / False
        for tile in wall_tiles:
            wall_lines = getSquareLines(tile, tile_size)
            for line in wall_lines:
                if intersect((pt1,pt2), line):
                    return True
        return False

def findAdjacentValues(a, v1, v2, marker_value):  
    # Frontiers is where free spaces meets unexplored space
    b = np.where(np.logical_or(a == v1, a == v2), a, 1000000)
    diff_b1 = np.diff(b, axis=0)
    diff_b2 = np.diff(b, axis=1)
    
    c1 = np.where(appendZeroRow(diff_b1) == v1-v2, 1, 0)
    c2 = np.where(appendZeroColumn(diff_b2) == v1-v2, 1, 0)
    c3 = np.where(preappendZeroRow(diff_b1) == v2-v1, 1, 0)
    c4 = np.where(preappendZeroColumn(diff_b2) == v2-v1, 1, 0)
    
    d1 = np.bitwise_or(c1, c2)
    d2 = np.bitwise_or(c3, c4)
    e = np.bitwise_or(d1, d2)
    f = np.where(e == 1, marker_value, a)
    return f

def appendZeroRow(a):
    return np.vstack((a, np.zeros((1,a.shape[1]))))
    
def appendZeroColumn(a):
    return np.hstack((a, np.zeros((a.shape[0],1))))

def preappendZeroRow(a):
    return np.vstack((np.zeros((1,a.shape[1])), a))
    
def preappendZeroColumn(a):
    return np.hstack((np.zeros((a.shape[0],1)), a))

def make2DListInt(list2d):
    list2d_int = []
    for list1d in list2d:
        list1d_int = []
        for element in list1d:
            list1d_int.append(int(element))
        list2d_int.append(list1d_int)
        
    return list2d_int

def getVisibleTiles(pos, obstacle_map, sensor_range, rays):
    # Obstacles are marked with 1 and free space is marked with 0
    visible_tiles = []
    num_tiles = obstacle_map.shape
    mod = np.array([pos[0] - sensor_range, pos[1] - sensor_range])
    for ray in rays:
        for ray_component in ray:
            tile_pos = (ray_component[0]+mod[0], ray_component[1]+mod[1])
            # Make sure the tile is inside the bounds of the env
            if tile_pos[0] < num_tiles[0] and tile_pos[1] < num_tiles[1]:
                if tile_pos[0] >= 0 and tile_pos[1] >= 0:
                    visible_tiles.append(tile_pos)
                    if obstacle_map[tile_pos] == 1:
                        break
    return visible_tiles

def getIndices(pt1, pt2):
    v1 = min(pt1[0], pt2[0])
    v2 = max(pt1[0], pt2[0]) + 1
    h1 = min(pt1[1], pt2[1])
    h2 = max(pt1[1], pt2[1]) + 1
    if v1 == v2:
        v2 += 1
    if h1 == h2:
        h2 += 1
    return (v1,v2,h1,h2)

def flattenedIndex2Touple(index, shape):
    return (index//shape[1], index%shape[1])

def getScaledRectangles(v1,v2,h1,h2, shortest_side_len):
    a1 = v2-v1
    a2 = h2-h1
    if a2 < a1:
        b2 = shortest_side_len
        b1 = int((a1*b2)/a2)
    else:
        b1 = shortest_side_len
        b2 = int((b1*a2)/a1)
        
    r1 = getIndices((v1,h1),(v1+b2, h1+b1))
    r2 = getIndices((v1,h2-b1),(v1+b2, h2))
    r3 = getIndices((v2-b2,h1),(v2, h1+b1))
    r4 = getIndices((v2-b2,h2-b1),(v2, h2))
    
    return (r1,r2,r3,r4)

def pointOnRectangleVertice(rs, v):
    print(v)
    print(rs)
    for r in rs:
        if v in r:
            return r
    return None