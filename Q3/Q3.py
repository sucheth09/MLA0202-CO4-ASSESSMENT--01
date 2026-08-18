import numpy as np
import networkx as nx

image = np.array([
    [100, 102, 250, 101],
    [98, 105, 103, 100],
    [101, 99, 104, 102],
    [100, 103, 105, 101]
], dtype=float)

rows, cols = image.shape

graph = nx.Graph()

for i in range(rows):
    for j in range(cols):
        graph.add_node((i, j), value=image[i, j])

for i in range(rows):
    for j in range(cols):
        if i + 1 < rows:
            graph.add_edge((i, j), (i + 1, j))
        if j + 1 < cols:
            graph.add_edge((i, j), (i, j + 1))

updated_image = image.copy()

for i in range(rows):
    for j in range(cols):
        neighbors = list(graph.neighbors((i, j)))

        if neighbors:
            neighbor_values = [
                image[x, y]
                for x, y in neighbors
            ]

            updated_image[i, j] = (
                image[i, j] + np.mean(neighbor_values)
            ) / 2

print("Original Image:")
print(image.astype(int))

print("\nUpdated Image:")
print(updated_image.astype(int))
