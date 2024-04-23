Introduzione
------------

Di seguito e' descritto il "protocollo" usato per implementare la "data distillation"
secondo le regole che ci siamo detti in queste settimane.


Nota: dalle prove fatte fino ad ora, NON SEMBRA che ci sia un comportamento significativamente differente
TRA una ricerca random dei parametri ed una ricerca usando un ottimizzatore bayesiano.

L'ottimizzatore bayesiano inizia con una generazione casuale dei parametri ma poi inizia ad usare
il progcesso gaussiano. Non ho ancora capito dopo QUANTO TEMPO/QUANTI STEP inizia con il GP, MA sono certo
che lo fa perche' ho messo un breakpoint nella classe coinvolta e dopo un po' "ci passa".
Il QUANTO TEMPO/QUANTI STEP dipende dai parametri passati alla classe.
Il problema e' che ci sono tanti parametri ma NON E" NECESSARIO assegnare a tutti iun valore perche'
sono stati gia' inizializzati con dei valori di "default" considerati ragionevoli dall'implementatore.
L'altro problema e' che i parametri piu' interessanti non sono indicati esplicitamente nella documentazione
MA bisogna dedurli analizzando il codice sorgente. E senza sapere nel dettaglio come funziona il tutto
e' un po' complicato capire dove andare a guardare.


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

       Una variante era quella di usare 'Xr,yr' per generare un GTC "locale".
       Ma con Jianyi abbiamo visto che questo non e' ragionevole perche'
       questo vorrebbe dire che, di volta in volta, ad 'Xd' verrebbero assegnate
       (potenzialmente) etichette diverse.
       Questo aumenta la robustezza del modello MA non e' piu 'deterministico'.
       Inoltre, in ogni momento dovrebbe essere possibile, dati 'Xd' generare
       le etichette 'yd' anche AL DI FUORI del processo di 'model selection'

    5) quando viene richiesta la predizione ('predict(Xr)' ) o la valutazione
       dello score usando 'score(Xr, yr)', che non e' altro che

            measure_score(y_true, predict(Xr))

       usa il classificatore 'distillato 'DC' per fare la predizione

            y_pred = DC.pred(Xr)

A questo punto l'"estimatore" puo' venir usato dal "model selector" per
trovare i "parametri" migliori (i nostri D*R punti nello spazio distillato).

Il "model selector" funziona nel seguente modeo":

    1) almeno inizialmente genera dei parametri random

    2) crea l'"estimatore" con i parametri cosi' generati

    3) usa la cross validation per valutare l'"estimatore" usando
       'X,'y' passati al metodo 'fit(X,Y)' ('interfaccia del "model selector"
       e' la stessa di tutti gli estimatori della libreria scikit-learn
       e librerie compatibili)
       I 'X,y' per il "model selector" da usare sono, ovviamente, 'Xr,yr',
       cioe' il dataset originale nello spazio ridotto

    4) il "model selector" suddivide X e y in CV parti, e, a rotazione,
       usa (CV-1) per il training e 1 per la valutazione delle performace (lo score)
       Nota: in teoria la CV non sarebbe necessaria, MA al momento
       le librerie lo richiedono e non c'e' un modo "ufficiale"
       per evitarlo.
       Per fare questo, il "model selector" DOPO aver create l'"estimatore" con
       i parametri scelti, chiama il corrispondente metodo 'fit(X,Y)' (del NOSTRO
       estimatore)

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


Dopo che si fa?
---------------

Alla fine del passo precedente, abbiamo i "parametri migliori" e cioe l'insieme dei
D punti nello spazio ridotto (R dimension) 'Xd' e, usando il GTC addestrato inizialmente,
siamo sempre in grado di assegnare loro le etichette.

Problema: questi, pero' sono punti che stanno in uno spazio continuo ridotto MENTRE a noi
servono dei punti con TUTTE coordinate categoriche nello spazio originale.

Ricordiamo che il dataset nello spazio ridotto e' stato creato nel seguente modeo:

    1) dataset con colonne categoriche
    2) trasformazione colonne categorcihe mediante oneho-encoding. transformazione INVERTIBILE
    3) dimensional reduction. PROBLEMA: la trasformazione NON E' INVERTIBILE

Quindi non c'e' modo di passare dallo spazio ridotto allo spazio originario. Inoltre, anche potendolo fare
I passi sarebbero

    3')  dimensional expansion: dallo spazio ridotto si arriva allo spazio {0,1}^n.
         MA mentre le coordinale del dataset originale eroano SOLO 0,1, in questo caso
         le coordinate sono dei valori continui TRA [0,1]. Quindi COMUNQUE c'e' un problema
         di "discretizzazione" perche' dobbiamo ricostruire una rappresentazione "onehot"
    2')  dalla rappresentazione 'onehot' "sintetica", ricostruire la rappresentazione categorica

Quindi, poiche' "comunque" c'e' un problema di rappresentazione, un'approccio alternativo potrebbe essere questo:

    1) per ogni punto distillato nello spazio ridotto, si va a cercare quale sia il punto, SEMPRE nello spazio
       ridotto, del dataset originale.

    2) trovato il punto (MA c'e' ne potrebbero essere diversi alla STESSA distanza, al momento prendiamo il PRIMO)
       ricuperiamo anche la corrispondente etichetta CHE, ovviamente, potrebbe essere DIVERSA dal'etichetta
       assegnata al punto distillato

    3) a questo punto abbiamo convertito i punti distillati (e relative etichette) in punti del dataset originario
       (e relative etichette) SEMPRE nello spazio ridotto.
       PROBLEMA: non e' detto che tali punti siano TUTTI diversi. Ci possono essere (e ci sono) duplicazioni,
       quindi risulta necessario eliminarle.

    4) sapendo quali punti originali nello spazio ridotto sono stati identificati, risulta immediato ricuperare
       i punti originali nello spazio originale (categorico).

Ed ora il passo successivo: dobbiamo VALUTARE la qualita' del dataset cosi' trovato (il "coreset")


    1) trasformazione dei punti del dataset distillato nello spazio ridotto
    2) addestramento del DC (classificatore distillato)
    3) valutazione delle performance di DC sull'intero dataset originario.


Si puo' far di meglio?
----------------------

    SE vogliamo un datset distillato NELLO SPAZIO originale (categorico) mi sa di no.

    MA SE ci bastano i punti nello spazio ridotto, potremmo limitarci ai punti stessi, MA
    bisogna anche portarsi dietro gli oggetti usati per le trasformazioni per poter
    esser sicuri di fare SEMPRE le stesse operazioni:

        1) i trasformator "categorico-> onehot" per X E y nello spazio categorico
        2) il "dimensional reductor" da alta dimensionalita' a bassa dimensionalita'
        3) 'Xr,yr' i punti distillati (e relative etichette) nello spazio ridotto
        3) il classificatore distillato "DC"

    In teoria, il classificatore distillato "DC" non e' necessario perche' usando 'Xr,yr'
    ed inizializzando nello stesso identico modo gli eventuali generatori di numeri casiali usati nel modello,
    e' sempre possibile "ricostruire" il classificatore da zero.
    Pero' se c'e' lo salviamo, e' meglio.
