import cv2
import numpy as np

# 1. Carica l'immagine di partenza
img = cv2.imread('tua_immagine.jpg')
h, w = img.shape[:2]

# 2. Definisci il centro dell'effetto e la forza della distorsione
center_x, center_y = w / 2, h / 2
radius = max(w, h) / 2
distortion_strength = 2.0  # Più alto è il valore, più forte è l'effetto bolla

# 3. Crea le griglie di coordinate per il remap
y, x = np.indices((h, w), dtype=np.float32)

# Normalizza le coordinate rispetto al centro e scala per il raggio
dx = (x - center_x) / radius
dy = (y - center_y) / radius
distance = np.sqrt(dx**2 + dy**2)

# 4. Applica la formula di distorsione a barilotto (fish-eye)
# Puoi modificare la funzione trigonometrica per regolare la curvatura
mask = distance < 1.0
factor = np.zeros_like(distance)
factor[mask] = np.sin(distance[mask] * np.pi / 2) ** (1.0 / distortion_strength) / distance[mask]
factor[~mask] = 1.0

# Calcola le nuove coordinate dei pixel
map_x = x * factor * (1 - mask) + x * mask # semplificato per il mapping inverso
map_x = center_x + dx * factor * radius
map_y = center_y + dy * factor * radius

# 5. Esegui il rimappaggio dell'immagine
output_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Salva o mostra il risultato
cv2.imwrite('risultato_fisheye.jpg', output_img)