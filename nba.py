import numpy as np
#import seaborn as sns
#import sklearn.cluster as cluster
import time
from nba_api.stats.endpoints import franchiseplayers
from nba_api.stats.static import teams
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import math

def dist(a,b):
    dist = 0
    for i in range(len(a)):
        dist += (a[i] - b[i])**2
    return np.sqrt(dist)

def three_d_dist(a,b):
    dist = 0
    for i in range(len(a)):
        pt_dist = 0
        pt_a = a[i]
        pt_b = b[i]
        for j in range(len(pt_a)):
            pt_dist += (pt_a[j] - pt_b[j])**2
        dist += pt_dist
    return dist

# Find teams by full name.
bulls = teams.find_teams_by_full_name('bull')
bulls_id = bulls[0]['id']
#print(bulls_id)

bulls_franchise_players = franchiseplayers.FranchisePlayers(bulls_id).get_data_frames()[0]
#for player in bulls_franchise_players:
#    print(bulls_franchise_players[player])
pts = bulls_franchise_players['PTS']
reb = bulls_franchise_players['REB']
ast = bulls_franchise_players['AST']
ts = pd.DataFrame({
    'pts':pts,
    'reb':reb,
    'ast':ast,
})

data = np.transpose(np.array((pts,reb,ast)))

pts_lim = 30000
reb_lim = 6000
ast_lim = 5000
upper_lims = [pts_lim, reb_lim, ast_lim]
lower_lims = [0,0,0]
cluster_locs = [(0,0,0)]*5
clusters = [0]*len(pts)
#print(range(len(clusters)))
for i in range(len(cluster_locs)):
    cluster_locs[i] = np.random.uniform(lower_lims, upper_lims, 3)
#print('first cluster_locs: ', cluster_locs)
cluster_diff = float('inf')
while cluster_diff > 10:
    # assign each point to its closest cluster
    for j in range(len(data)):
        data_pt = data[j]
        pt_dists = [0] * 5
        for i in range(len(cluster_locs)):
            cluster_loc = cluster_locs[i]
            pt_dists[i] = dist(data_pt, cluster_loc)
        #print('pt_dists: ', pt_dists)
        clusters[j] = np.argmin(pt_dists)
    #print('clusters:', clusters)

    new_clusters = [(0,0,0)]*5
    cluster_cnts = [(0,0,0)]*5
    # compute new centroids from current clusters
    for i in range(len(clusters)):
        data_pt_cluster = clusters[i]
        new_clusters[data_pt_cluster] = [sum(x) for x in zip(new_clusters[data_pt_cluster],data[i])]
        cluster_cnts[data_pt_cluster] = [sum(x) for x in zip(cluster_cnts[data_pt_cluster], (1,1,1))]
    new_clusters = np.divide(new_clusters, cluster_cnts)
    #print('new_clusters before ', new_clusters)
    for i in range(len(new_clusters)):
        new_cluster = new_clusters[i]
        if pd.isna(new_cluster[0]):
            #print('nan detected')
            new_clusters[i] = [0,0,0]
    #print('new_clusters after ', new_clusters)

    cluster_diff = three_d_dist(new_clusters, cluster_locs)
    cluster_locs = new_clusters
    print('cluster_diff: ', cluster_diff)
    #print('second cluster locs: ', cluster_locs)

print('cluster_locs: ', cluster_locs)

threedee = plt.figure().gca(projection='3d')
for i in range(len(data)):
    data_pt = data[i]
    data_pt_cluster = clusters[i]
    if data_pt_cluster == 0:
        threedee.scatter(data_pt[0],data_pt[1],data_pt[2], c='b')
    elif data_pt_cluster == 1:
        threedee.scatter(data_pt[0],data_pt[1],data_pt[2], c='r')
    elif data_pt_cluster == 2:
        threedee.scatter(data_pt[0],data_pt[1],data_pt[2], c='g')
    elif data_pt_cluster == 3:
        threedee.scatter(data_pt[0],data_pt[1],data_pt[2], c='y')
    else:
        threedee.scatter(data_pt[0], data_pt[1], data_pt[2], c='m')



#print('data:', data)



#threedee.scatter(pts, reb, ast)
threedee.set_xlabel('Points')
threedee.set_ylabel('Rebounds')
threedee.set_zlabel('Assists')
plt.show()




