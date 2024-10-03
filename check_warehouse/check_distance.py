from math import sin, cos, atan2, sqrt, asin, radians
import haversine as hs

def sq(x): return x*x


def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # R = 6373.0
    R = 6371.0088
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sq(sin(dlat/2)) + cos(lat1) * cos(lat2) * sq(sin(dlon/2))
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    # c = 2 * asin(sqrt(a))
    return R * c

def haversine(pt1, pt2):
    return distance(*pt1, *pt2)


# latitude, longitude

loc1=(28.426846,77.088834)
loc2=(28.394231,77.050308)
print(haversine(loc1,loc2))
print(hs.haversine(loc1,loc2))

loc1=(19.0760, 72.8777)
loc2=(18.5204, 73.8567)
print(haversine(loc1,loc2))
print(hs.haversine(loc1,loc2))
