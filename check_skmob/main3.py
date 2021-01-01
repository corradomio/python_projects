import folium
import matplotlib.pyplot as plt

m = folium.Map(
    location=[45.5236, -122.6750],
    zoom_start=13
)
""":type: folium.folium.Map"""

m.save('index.html')

folium.Map(
    location=[45.5236, -122.6750],
    tiles='Stamen Toner',
    zoom_start=13
)

