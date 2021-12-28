import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def networkxTest():
    G = nx.path_graph(6)
    center_nodes = {0, 3}
    cells = nx.voronoi_cells(G, center_nodes)
    partition = set(map(frozenset, cells.values()))
    print(cells)
    print(partition)
    print(sorted(map(sorted, partition)))

    #nx.draw(G)
    #plt.show()

def voronoiTest():
    points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)
    plt.show()
    
voronoiTest()