Supponiamo un problema di classificazione binaria (con valori {0,1})

Sia X una matrice NxM con valori reali in [0,1]. 
    Per dare dei numeri, supponiamo: N=100, M=2
    Uso M=2 per essere sicuro di ragionare in termini di "matrici" e non di vettori.
    In questo modo, se uso M=1 o M=1000, il ragionamento non cambia.

Sia y un vettore di N interi in {0,1} (le due categorie)

1) addestro GTC, il classificatore Ground Truth con (X, y)

Ora, supponiamo di voler trovare K punti, diciamo K=10, da usare per addestrare
un classificatore DC, dello stesso tipo di GTC. Gli "iperparametri" di DC sono
questi K punti, ma poiche' ogni punto e' composto da M dimensioni, il numero
totale di iperparametri di DC e' K*M. In questo caso 10*2=20.

Quindi, posso considerare, come spazio degli iperparametri di DC lo spazio [0,1]^K*M

Il protocollo da seguire e' il seguente:

1) genero un punto in [0,1]^KxM: 'Pd'

2) converto 'Pd' in una matrice KxM, cioe' in K punti in [0,1]^M, che chiamo 'Xd'

3) uso GTC per assegnare le etichette a questi K punti: 'yd'
   A questo punto ho un "dataset distillato": (Xd, yd)

4) creo un nuovo DC e lo addestro con il "dataset distillato" (Xd, yd)

5) applico DC a 'X' per ottenere 'yp', le predizioni sulle etichette fornite da DC
   sul dataset originario 'X'

6) calcolo l'accuracy: 'Ad=accuracy(y, yp)'.
   A questo punto abbiamo il 'punto' da usare con l'ottimizzatore bayesiano: (Xd, Ad)
   (piu' precisamente:  (Pd, Ad))

7) aggiorno l'ottimizzatore con il nuovo punto (Xd, Ad)

8) chiedo all'ottimizzatore di propormi un'altro punto, 'Pd_prossimo'

9) se ho raggiunto la condizione di termine elaborazione, STOP,
   altrimenti uso 'Pd_prossimo' come 'Pd' e goto 2)


