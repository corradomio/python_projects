data distillation / core set
----------------------------

Il protocollo utilizzato e' il seguente:

1) caricato il dataset con tutte le colonne categoriche (anche se ci dovesse essere solo un sottoinsieme di
   colonne di tipo categorico, il protocollo sarebbe lo stesso
   8124 righe

2) split dataset in X, y
   X:22 colonne (tutte categoriche), y:1 colonna (boolean)

3) conversione delle colonne categoriche mediante onehot encoding.
   X:112 colonne, y:1 colonna (0,1)

4) dimensional reduction a 'R'=7 colonne. Usato 7 peche' con 8 UMAP si lamenta
   dicendo che in questo caso sarebbe meglio usare un Autoencoder
   X:7 colonne, y:1 colonna

5) addestramento di un classificatore usando il dataset nello spazio ridotto
   con la solita tecnica train/test.
   Questo classificatore (GTC: Ground Trith Classifier) verra' utilizzato
   per assegnare le etichette ai punti sintetici.


Ora si devono cercare 'D' (distilled) punti sintetici (nello spazio ridotto)
che possono essere usati per addestrare un classificatore. Scelto:

    D = 100
    classificatore: DecisionTreeClassifier

Quindi, poiche' servono 'D'(=100) punti, ed ogni punto e' definito da 'R'(=7) coordinate
numeriche, servono 'D*R'(=700) valori numerici.

Definiamo un "estimatore" 'E' (che USA dei classificatori per fare le sue cose ma che, tecnicamente,
NON E' un classificatoe, che e' parametrizzato da 'D*R' parametri PIU' 1 parametro
di servizio che non e' altro che un oggetto che fornisce servizi aggiuntivi
(poi vediamo quali)

Servono alcune cose:

    1) un oggetto "Bounds" che ha il compito di 'sapere' quali sono i possibili valori
       che puo' assumere ogni parametro (ed eventualmente anche una distribuzione di
       probabilita')
       Supponiamo che la distribuzione sia uniforme, e di ricuperare il minimo ed il massimo
       di ogni colonna (7 colonne) della versione in bassa dimensionalita del dataset originale
       Supponiamo anche che i punti distillati dovranno rimanere necessariamente all'intero
       di questi range.
       Quindi, Bounds non sara altro che questi 7 range replicati 100 volte
    2) un generatore di parametri, partendo da Bouns. Questo perche', almeno all'inizio
       e' necessario generare un po' di punti a caso nel dominio dei parametri

    3) un "model selector" che ha il compito di cercare i parametri "migliori" per il
       nostro estimatore.
       Le librerie a disposizione (scikit-learn & sciki-optimize) mettono a disposizione
       due oggetti:

            "BayesSearchCV" che usa un ottimizzatore gaussiano
            "RandomSearchCV" che fa una ricerca random tra tutte le possibili combinazioni
                          di parametri

       Nota:  RandomSearchCV funziona SOLO con parametri in forma discreta e PRIMA crea
       il prodotto cartesiano di TUTTE le possibili combinazioni di parametri.
       Poiche' nel nostro caso tale numero sarebbe esagerato (i 4*10^1402) di cui parlavo
       tempo fa) ho fatto una implementazione ad hoc che genera i parametri in modo random
       "su richiesta"

Ora, il "model selector" funziona in questo modo:

    1) riceve in ingresso "Bounds", i possibili valori per OGNI parametro
    2) e l'"estimatore" per il quale dobbiamo trovare i parametri "migliori"


Come funziona il nostro "estimatore"
------------------------------------

L'"estimatore" ha come parametri i 'D*R' (=700) valore corrispondenti alle R coordinate
dei D pointi:

    1) prende i parametri e li organizza come una matrice di feature 'Xd' (X distilled)
       di D righe e R colonne
    2) usa il GTC, precedentemente creato, per generare le etichette 'yd'
    3) crea un 'DC' (Distilled Classifier) addestrandolo con 'Xd,yd'

    4) quando riceve in ingresso 'Xr,yr' (nello spazio ridotto) per il training
       (metodo 'fit(X,y)' dell'interfaccia scikit-learn) non fa niente.
       Una variante era quella di usare Xr, yr per generare un GTC "locale".
       Ma con Jianyi abbiamo visto che questo non e' ragionevole perche'
       questo vorrebbe dire che, di volta in volta, ad 'Xd' verrebbero assegnate
       (potenzialmente) etichette diverse.
       QUesto aumenta la robustezza del modello MA non e' piu 'deterministico'

    5) quando viene richiesta la predizione ('predict(Xr)' ) o la valutazione
       dello score usando 'score(Xr, yr)', che non e' altro che

            score(y_true, predict(Xr))

       usa il classificatore 'distillato 'DC' per fare la predizione

            y_pred = DC.pred(Xr)

A questo punto puo' venir usato dal "model selector" il quale:

    1) almeno inizialmente genera dei parametri random

    2) crea l'"estimatore" con i parametri cosi' generati

    3) usa la cross validation per valutare l'"estimatore" usando
       'X,'y' passati al metodo 'fit(X,Y)' (l'interfaccia e' la stessa
       di tutti gli estimatore ANCHE per il "model selector"

    4) suddivide X e y in CV parti, e, a rotazione, usa (CV-1) per
       il training e 1 per la valutazione delle performace (lo score)
       Nota: in teoria la CV non sarebbe necessaria, MA al momento
       le librerie lo richiedono e non c'e' un modo "ufficiale"
       per evitarlo.

    5) fa le medie degli score dello stesso "estimatore" (perche'
       usa gli STESSI parametri) con diversi set di train & validazione
       e si salva parametri e score

    6) quindi ricomincia il processo da 1)

Il model selector "RandomSearchCV" usa SOLO parametri generati random

Il model selector "BayesSearchCV" inizia per un po' con parametri generati
random, ma dopo un po' inizia ad usare i processi gaussiani per trovare
quali siano i parametri migliori.

Nota: dopo quanto tempo non mi e' ancora chiaro.
Il problema e' che ci sono TANTI parametri e non e' necessario specificarli
tutti perche' hanno gia' dei valori di default 'ragionevoli' almeno secondo
l'implementatore.