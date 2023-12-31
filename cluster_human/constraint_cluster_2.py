import pickle
from sklearn.cluster import AgglomerativeClustering

with open('animal_list.pkl', 'rb') as f:
    animals = pickle.load(f)

# iteration 0
cosine_distances = [
    [0.000, 0.547, 0.539, 0.301, 0.436, 0.462, 0.424, 0.500, 0.451],
    [0.547, 0.000, 0.314, 0.549, 0.333, 0.365, 0.553, 0.541, 0.294],
    [0.539, 0.314, 0.000, 0.390, 0.226, 0.261, 0.434, 0.549, 0.253],
    [0.301, 0.549, 0.390, 0.000, 0.334, 0.469, 0.337, 0.443, 0.439],
    [0.436, 0.333, 0.226, 0.334, 0.000, 0.231, 0.415, 0.500, 0.175],
    [0.462, 0.365, 0.261, 0.469, 0.231, 0.000, 0.469, 0.485, 0.274],
    [0.424, 0.553, 0.434, 0.337, 0.415, 0.469, 0.000, 0.175, 0.468],
    [0.500, 0.541, 0.549, 0.443, 0.500, 0.485, 0.175, 0.000, 0.483],
    [0.451, 0.294, 0.253, 0.439, 0.175, 0.274, 0.468, 0.483, 0.000]
]

# iteration 1
cosine_distances = [
    [0.000, 1.000, 0.539, 0.301, 0.436, 0.462, 0.424, 0.500, 0.451],
    [1.000, 0.000, 0.000, 0.549, 1.000, 1.000, 0.553, 0.541, 1.000],
    [0.539, 0.000, 0.000, 0.390, 1.000, 1.000, 0.434, 0.549, 1.000],
    [0.301, 0.549, 0.390, 0.000, 0.334, 0.469, 0.337, 0.443, 0.439],
    [0.436, 1.000, 1.000, 0.334, 0.000, 0.231, 0.415, 0.500, 0.175],
    [0.462, 1.000, 1.000, 0.469, 0.231, 0.000, 0.469, 0.485, 0.274],
    [0.424, 0.553, 0.434, 0.337, 0.415, 0.469, 0.000, 0.000, 0.468],
    [0.500, 0.541, 0.549, 0.443, 0.500, 0.485, 0.000, 0.000, 0.483],
    [0.451, 1.000, 1.000, 0.439, 0.175, 0.274, 0.468, 0.483, 0.000]
]

# iteration 2
cosine_distances = [
    [0.000, 1.000, 0.539, 1.000, 1.000, 1.000, 0.424, 0.500, 0.451],
    [1.000, 0.000, 0.000, 0.549, 1.000, 1.000, 0.553, 0.541, 1.000],
    [0.539, 0.000, 0.000, 0.390, 1.000, 1.000, 0.434, 0.549, 1.000],
    [1.000, 0.549, 0.390, 0.000, 0.000, 0.000, 0.337, 0.443, 1.000],
    [1.000, 1.000, 1.000, 0.000, 0.000, 0.000, 0.415, 0.500, 1.000],
    [1.000, 1.000, 1.000, 0.000, 0.000, 0.000, 0.469, 0.485, 1.000],
    [0.424, 0.553, 0.434, 0.337, 0.415, 0.469, 0.000, 0.000, 0.468],
    [0.500, 0.541, 0.549, 0.443, 0.500, 0.485, 0.000, 0.000, 0.483],
    [0.451, 1.000, 1.000, 1.000, 1.000, 1.000, 0.468, 0.483, 0.000]
]

# iteration 3
cosine_distances = [
    [0.000, 1.000, 0.539, 1.000, 1.000, 1.000, 0.424, 0.500, 1.000],
    [1.000, 0.000, 0.000, 0.549, 1.000, 1.000, 0.553, 0.541, 1.000],
    [0.539, 0.000, 0.000, 0.390, 1.000, 1.000, 0.434, 0.549, 1.000],
    [1.000, 0.549, 0.390, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000],
    [1.000, 1.000, 1.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000],
    [1.000, 1.000, 1.000, 0.000, 0.000, 0.000, 1.000, 1.000, 1.000],
    [0.424, 0.553, 0.434, 1.000, 1.000, 1.000, 0.000, 0.000, 0.468],
    [0.500, 0.541, 0.549, 1.000, 1.000, 1.000, 0.000, 0.000, 0.483],
    [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.468, 0.483, 0.000]
]

# with open('animal_cos_dist.pkl', 'rb') as f:
#     cosine_distances = pickle.load(f)
# print(cosine_distances)

# Step 1: Create an AgglomerativeClustering model
clustering_model = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='complete')

# Step 2: Fit the model
clustering_model.fit(cosine_distances)

# Step 3: Retrieve cluster labels
cluster_labels = clustering_model.labels_

# Step 4: Interpret and present the results
clusters = {}
for i, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(animals[i])

i = 1
for cluster, members in clusters.items():
    print(f"Cluster {i}: {' '.join(members)}")
    i += 1