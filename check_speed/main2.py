import time
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from win32comext.adsi.demos.scp import verbose

# Verifica se la GPU è disponibile
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# inserisci qui sotto le coordinate: *
X = np.array([    66.343,    53.393,    38.528,    23.545,    10.252,    0.254,    -5.241,    -5.569,    -0.691,    8.803,    21.768,    36.638,    51.617,    58.826,    92.284,    91.783,    91.282,    78.598,    73.552,    77.882,    89.727,    89.226,    88.724,    88.223,    87.722,    87.221,    74.263,    73.619,    85.2,    84.199,    83.197,    82.195,    73.553,    80.423,    79.421,    78.169,    76.668,    62.632,    53.608,    52.022,    58.303,    70.759,    86.041,    100.037,    108.983,    110.473,    104.105,    91.592,    93.094,    96.974,    94.388,    95.389,    103.96,    97.161,    98.163,    99.165,    100.167,    111.04,    115.835,    112.561,    102.212,    102.713,    103.215,    103.716,    104.217,    104.718,    117.137,    122.42,    118.345,    106.273,    106.775,    107.276,    112.9,    122.962,    133.024,    143.085,    153.147,    163.209,    178.318,    175.11,    158.225,    152.084,    142.022,    131.961,    121.899,    111.837,    101.775,    94.327,    71.98,    64.772,    ])
Y = np.array([    46,    84,    88,    87,    22,    6,    45,    54,    83,    84,    51,    5,    70,    87,    42,    52,    75,    45,    20,    9,    10,    75,    56,    89,    39,    13,    52,    41,    78,    47,    61,    41,    4,    29,    69,    60,    67,    10,    86,    40,    12,    29,    79,    88,    68,    9,    62,    10,    17,    65,    88,    20,    52,    44,    64,    18,    71,    75,    21,    64,    36,    6,    84,    33,    18,    79,    53,    54,    83,    26,    85,    27,    36,    22,    87,    47,    60,    46,    12,    29,    13,    8,    69,    70,    10,    24,    29,    31,    62,    39,    ])


# ------------------------------ *

# Imposta il seme per Python
random.seed(45)

# Imposta il seme per NumPy
np.random.seed(45)

# Imposta il seme per TensorFlow
tf.random.set_seed(45)

# Inizia il timer
start_time = time.time()

# Ordina i dati in base ai valori di X
sorted_indices = np.argsort(X, axis=0).flatten()
X_sorted = X[sorted_indices]
Y_sorted = Y[sorted_indices]

# Suddivisione dei dati in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X_sorted, Y_sorted, test_size=0.2, random_state=45)

# Assicurati che X_train e X_test siano array 2D
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Standardizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creazione del modello di rete neurale
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Layer di output
])

# Compilazione del modello con un ottimizzatore adattivo
model.compile(optimizer='Adam', loss='mean_squared_error')

# Addestramento del modello
model.fit(X_train, y_train, epochs=1000, batch_size=10, validation_split=0.1, verbose=0)

# Valutazione del modello
loss = model.evaluate(X_test, y_test)
print(f'Loss sul set di test: {loss}')

# Effettuare delle previsioni
predictions = model.predict(X_test)
print(f'Prime 5 previsioni: {predictions[:5].flatten()}')  # Flatten per rendere il formato 1D

# Calcolo dei residui
residui = y_test - predictions.flatten()
print(f'Prime 5 residui: {residui[:5]}')

# Calcolo di RMS e RMSE
rms = np.sqrt(np.mean(np.square(y_test)))
rmse = np.sqrt(loss)

print(f'RMS: {rms}')
print(f'RMSE: {rmse}')

# Funzione per filtrare le previsioni in base all'intervallo di y_test
def filter_by_range(predictions, y_test, lower_bound, upper_bound):
    filtered_indices = (y_test >= lower_bound) & (y_test <= upper_bound)
    filtered_predictions = predictions[filtered_indices]
    filtered_y_test = y_test[filtered_indices]
    return filtered_predictions, filtered_y_test

# Stampa i valori di y_test e delle previsioni prima del filtro
print(f'Valori di y_test: {y_test}')
print(f'Previsioni: {predictions.flatten()}')

# Esempio di utilizzo della funzione di filtro
lower_bound = 46
upper_bound = 90
filtered_predictions, filtered_y_test = filter_by_range(predictions, y_test, lower_bound, upper_bound)

# Calcolo dei residui per le previsioni filtrate
filtered_residui = filtered_y_test - filtered_predictions.flatten()

# Stampa dei risultati filtrati in verticale con spazi di riga
print("\nPrevisioni filtrate (da {} a {}):".format(lower_bound, upper_bound))
for pred, real, resid in zip(filtered_predictions.flatten(), filtered_y_test, filtered_residui):
    print(f"Previsione: {pred}, Valore reale: {real}, Residuo: {resid}\n")

# Stampa delle prime 5 previsioni e residui in verticale con spazi di riga
print("\nPrime 5 previsioni e residui:")
for i in range(5):
    print(f"Previsione: {predictions[i].flatten()[0]}, Residuo: {residui[i]}\n")

# Ferma il timer e stampa il tempo di esecuzione
end_time = time.time()
execution_time = end_time - start_time
print(f'Tempo di esecuzione: {execution_time} secondi')
