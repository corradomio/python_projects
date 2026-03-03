X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import RadiusNeighborsRegressor
neigh = RadiusNeighborsRegressor(radius=1.0)
neigh.fit(X, y)
print(neigh.predict([[1.5]]))
