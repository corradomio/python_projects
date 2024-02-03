software decomposition (~colorazione grafo)
    rileggere l'articolo/revisione testo/ultime sezioni
    controllare il body dell'articolo
    dalla 3 sezione in poi

    IEEE Transactions on Services Computing (TSC) 
    bio/biography


-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------

Special Issue: articolo nuovo
    elaborazione dati
    tesi corrado, ridurre la parte di ricerca, sottometterla
    multi view learning/co-training


preparazione del dato
    data distillitaion (classificazione!/regressione?)
        data pruning    tenendosi solo la frontiera
        data creation   per creare i dati mancanti

-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------

GUO Dataset pruning

approccio tipo active learning
bayesian optimization/gausian processes

generazione dati variazionale

-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------

statistica non parametrica
    All of Statistics - 2004
    All of Nonparametric Statistics - 2006

regressione non parametrica/kernel density

target learning: framework machine learning per fare delle stime
    influence functions

-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
A Tutorial on Bayesian Optimization
https://arxiv.org/pdf/1807.02811.pdf

Gaussian Processes for Machine Learning
https://gaussianprocess.org/gpml/chapters/RW.pdf

Bayesian Optimization 
https://bayesoptbook.com/book/bayesoptbook.pdf


-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
Uso di embedding per lavorare su un ogetto discretto
L'emedding converte lo spazio discreto in uno continuo
Ottimizzaizone boolean sul continuo
Inverso sul discreto


dataset
decision tree
ridurre il dataset
dataset -> array di 128 variabili booleane


datas pruning (data distillation e' un supercaso)


-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
discreto -> continuo
    proiezioni casuali (modelli lineari)
    linear embedding
    random projection embedding
    bayesian optimization random embedding


ottimizazione bayesiana su reticoli combinatori
    ottimo del livello 10 su 100

metodo COMBO



-----------------------------------------------------------------------------
- 
-----------------------------------------------------------------------------
core set selection (survey??)
    dipende dal task

POI distillation (survey??)
    
  
-----------------------------------------------------------------------------
- Whatsapp
-----------------------------------------------------------------------------

[Gabriele]
ciao, abbiamo due aspetti da sviluppare Bayesian Optimization e Data Distillation. Dell'aspetto Ottimizzazione Bayesiana abbiamo già detto (ho condiviso anche a Gianni la cartella /Gabriele_corrado); riguardo alla Distillation, io intendevo focalizzarmi sul sotto-problema della "coreset selection" (mentre la distillation comprende anche la creazione di esempi rappresentativi artefatti, la coreset selection si limita, nella sua accezione restrittiva a selezionare solo esempi già esistenti nel dataset originario). Ci sono un sacco di articoli in giro con vari obiettivi e differenze nelle definizioni (più o meno ampie), ma credo di avere trovato il nostro bandolo della matassa: si tratta dell'articolo "DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning" di Guo et al., che hanno raccolto e implementato anche con PyTorch tutti i metodi principali e forniscono anche un sacco di benchmark, così da consentirci di avere già un ampio spettro di termini di confronto. Altro articolo che fornisce dei benchmark dataset (ma non codice) è "DC-BENCH: Dataset Condensation Benchmark" di Cui ed al. Li salvo entrambi nella cartella condivisa.


iniziando con deep core si ha una visione del problema della coreset selection, in parallelo o separatamente uno può guardare per la bayesian optimization la molto figurativa introduzione fornita dal libro di Nguyen e l'a chiarissima e formale presentazioni di Candelieri (entrambe queste ultime salvate adesso nella cartella di corrado)


ho aggiunto anche il libro del mio senior collega Archetti, il cui capitolo 6 presenta un bel po' di strumenti software per la Bayesian optimization

[Gabriele]
Ho salvato nella cartella un altro papiro (che avevo già mostrato lunedì): "Efficient Black-Box Combinatorial Optimization" del 2020. Due cose vanno dette: 1) mi ha fregato l'idea di usare lo shapley/banzaf value per selezionare il subset (come avevamo fatto nella tesi di Corrado), infatti il papiro usa la base di Fourier invece della base di Moebius, quindi semplicemente codifica presenza assenza con +1 e -1 anziché con 1 e 0, ma strutturalmente è la stessa idea 2) la buona notizia è che non l'ha notato quasi nessuno, cioè il papiro non è citato da altri che ci costruiscano sopra, ma solo nei related work di lavori sulla black-box optimization. Quindi c'è spazio per costruirci sopra, riversando anche la nostra expertise in "Shapleyologia comparata" 😉


L'idea di fondo è questa: noi vogliamo ottimizzare una set function (che è black box). Ad ogni valutazione (dispendiosa) aggiorniamo il modello probabilistico (il surrogate model che adesso descrivo) e poi scegliamo il prossimo insieme usando un'acquisition function opportuna (ad esempio la Expected Improvement). Il modello probabilistico in Bayesian Optimization è una distribuzione di probabilità sulle possibili funzioni pseudo-booleane (con n punti candidati a far parte dell'insieme ottimale abbiamo funzioni 2^n-->R). Questo modello a rigore abiterebbe in uno spazio 2^n dimensionale, ma noi lo approssimiamo con lo Shapley value dei singoli punti, quindi lavoriamo in uno spazio n dimensionale, e se vogliamo anche usare l'interaction value aggiungiamo n(n-1)/2 dimensioni. Il resto è procedura standard (che estrinseco quando ci vediamo).


Va detto che noi non effettuiamo una ricerca sull'intero reticolo booleano, ma abbiamo come obiettivo quello di trovare un insieme di cardinalità k<<n, appunto il coreset; quindi possiamo usare il banzaf ridotto, come nella teso di Corrado (ecco un'altra novità rispetto al papiro salvato), ma con un numero di candidati n grande il calcolo è comunque proibitivo).


Adesso penso a come sfruttare un fatto nuovo rispetto alla tesi di Corrado, una "mossa magica": qui possiamo fare il training sull'intero campione di un modello di Machine Learning trasparente, come un Decision Tree, così da ottenere informazioni sulle frontiere tra le regioni e poi cercare i punti candidati proprio lì vicino, perché è lì che ci saranno i punti più informativi. Questa è la mi idea ad oggi, e si differenzia dal papiro salvato perché quello parla di ottimizzazione compibatoria generica, mentre qui noi abbiamo a che fare con un caso molto particolare, grazie alla "mossa magica".


La mossa magica può sembrare una mossa truffaldina, ma non lo è. Infatti l'obiettivo dell'ottimizzazione è ottenere un insieme (di k punti estratto degli n, ad esempio con con n=10k) sfruttando un certo budget di calcolo. Ora è noto che un round di training implica un effort direttamente proporzionale al numero di esempi utilizzato: indico il budget con b x n, dove b è il budget consumato per singolo esempio. Se mi danno un budget b x (2 n), io posso usare il budget b x n per la singola valutazione dell'intero training dataset, e 10 volte b k per valutare dieci insiemi candidati, ciascuno di k punti: quindi dopo la mossa magica ho 10 shot grazie ai quali valutare la funzione obiettivo (10 iterazioni dellla Bayesian Optimization). Questi numeri sono solo esemplificativi. Il concetto è che io posso spezzare il budget in modo ottimale tra mossa magica e numero di iterazioni consentite.


Questo da luogo ad un problema interessante, e cioè come si spezza in modo ottimale il budget.


Sembrerebbe un problema di ottimizzazione ad un terzo livello (forse semplice se la funzione associata al budget è convessa): il livello più esterno è ottimizzare il budget utilizzando campioni di dimensione ad esempio decrescente, poi usare l'ottimizzazione bayesiana per scegliere il campione di un certo livello k, poi all'interno di ogni singola iterazione dell'ottimizzazione bayesiana ottimizzare la funzione d'acquisizione (anche questo sembra un problema semplice).


Tutto ciò senza assumere alcuna struttura nello spazio dei dati di input. Ma se ciascun punto x = (x_1,...,x_D) appartiene ad uno spazio D-dimensionale (ad esempio Euclideo) si può cercare un embedding in uno spazio d-dimensionale con d<D (il che aiuta sicuramente la ricerca) anche senza ricorrere minimamente alla dispendiosa valutazione della funzione f(S).


Siccome il dataset da condensare contiene le coppie (x,y) (cioè le coppie (input, label)), l'embedding è particolarmente informativo e si sovrappone alla "mossa magica", cioè al primo round di classificazione che utilizza tutto il campione. Come sfruttare questo fatto, ad esempio con delle SVM, è da chiarire.


Ho inoltre un'osservazione riguardo a quanto avevo scritto sopra: mi è stata chiara fin da domenica, mentre viaggiavo sul flixbus per Lione, senza connessione internet, quindi la segnalo adesso che sono tornato e ho pututo rimettere la testa sulle cose di ricerca.


Mi scuso perché nella foga ho commesso una svista, per troppo pessimismo, quindi rimuovendola si guadagna.


Avevo detto che sostituivamo la funzione pseudo-booleana 2^n dimensionale con la sua approssimazione tramite Shapley/Banzaf (cioè l’approssimazione di grado 1) per cercare l’insieme che da il massimo di quest’ultima: ma ciò viene gratis, perché essendo quell’approssimazione additiva negli atomi, quando hai stimato lo shapley value di quelli, li metti in ordine e tieni i top k per costruire il “dream-team” dei migliori k elementi.
Quindi questa operazione si riduce al calcolo dello shapley value e non richiede ottimizzazione bayesiana.
Ma per l’approssimazione di grado 2 non si può usare questa scorciatoia.
Sappiamo che il problema si può formulare in termini di grafi (il second’ordine rappresenta l’interazione, diciamo grossomodo lo shapley interaction value) e che la sua soluzione è equivalente a un problema di min-cut o max-cut. Questo è un problema combinatorio degno dell’ottimizzazione Bayesiana.
Forse troviamo già qualcosa sul connubbio dei due. Però il nostro caso è speciale, perché noi non ci muoviamo alla cieca come abbiamo fatto nella feature selection.
Come dicevo possiamo contare su: 1) un embedding dello spazio X in cui i punti sono immersi, e anche dello spazio (X,Y), poi 2) sotto opportune ipotesi possiamo permetterci una mossa esplorativa in cui facciamo il training di un modello trasparente su tutti i punti.

sto elaborando un un documento word la procedura senza tutti gli ammennicoli che ho aggiunto qui sopra: ne verrà fuori un one-pager che condividerò


-----------------------------------------------------------------------------
- Discussione articolo
-----------------------------------------------------------------------------

Ricavare il coreset
Distillation: generare artefatti che aiutano l'algoritmo

1) Dimesnionality Reduction

2) mossa magica: Decision Tree per classificazione

    Riprodure: tasselazione dell'albero

3) scegliere dei punti che ricostruisce l'albero originale

    Mondrian: suddivisione 2D -> nD

    
KNN: punti prototipi (centroidi)

Discriminant Analisys

Come aiutare la BayesOpt usando euristiche che indicano quali zone sono piu' interessanti


-----------------------------------------------------------------------------

dataset R[n,d], B -> R[n*d] x B[n]    e' un punto singolo in R[n*d] con n etichette


generazione dataset:
    decidere il numero di dimensioni
    per ogni dimensione decidere il numero di suddivisioni (random [1,R])
    generare suddivisioni random in range [0,1] (o [-1, 1])
    assegnare etichette random ad ogni intersezione, oppure in modo che 
        "smart" in modo da non avere 2 celle adiacenti con lo stesso colore
        numero passi di suddivisioni DISPARI!
    generare dataset con categorie bilanciate

distillazione (NO coreset) -> coreset dai punti piu' vicini dopo la distillazione


modello pangloss (alternativa al federated learning -> centralized learning)


servono 2 classificatori (logistico):
    uno sul GT
    uno sul dataset distillato


budget numero di tentativi
accuracy non migliora piu' di tot


ottimizzazione 1 dimensionale


svm, mistura di gaussiane, polinomio
    modelli SEMPLICI (O(n))


acquisition function: come implementarla?
    modello: class dei polinomi


Expected improvement: che cosa ottimizzare con BayesOpt
Botorch: Bayesian Optimization Torch
--------------------------------

ottimizzazione bayesiana:
       min/max f(z)
         z in Z



dataset: 100x2
dataset:  10x2
classificazione: 2 categorie
    classificatore logistico
        metrica: accuracy


------------------------------------------------
aumento dimensione dataset e dimensioni
    100_000/10_000/1000
    100
    ..

dataset a dimensione piu' bassa rispetto

-----------------------------------------------------------------------------
- Whatsapp
-----------------------------------------------------------------------------

Una proiezione lineare da fare (da piccolo a grande) e' semplice: basta generare una matrice riempita dei numeri random.

MA (domandona): esiste un modo SEMPLICE per fare qualche proiezione NON lineare (anche se la non linearita' e di tipo "semplice")?

L'idea che mi era venuta e' quella di generare dei "segmenti di curva" (nello spazio grande), una per ogni asse dello spazio piccolo, e poi usare un po' di interpolazione per traformare le coordinate dei punti dallo spazio piccolo a quello grande.
Ma mi pare arzigogolato.
E poi SOLO se ne vale la pena.
Penso che una trasformazione lineare, per iniziare, dovrebbe essere gia' abbastanza



UNO
Supponiamo di avere tre classi R,G e B, giusto per fare un po’ di colore.
Le associo a tre segmenti che coprono l’intervallo [0,1] dell’asse t. Esiste una funzione semplice che mappa t in f(t) (la classe/colore). Adesso aggiungo una dimensione non informativa z, così che f(t,z)=f(t).

Dal punto di vista di un algoritmo come Isomap, t-sne, UMAP, questa roba può essere riproiettata su dimensionalità 1D facilmente. [questo è il ritornello 1D]

DUE
Adesso riparto da zero e adotto una variante, con trasformazione di coordinate e produco uno swiss-roll. Parto da una funzione semplice che mappa t in f(t) (la classe/colore).
Prendo la dimensione lineare t (su cui generare i punti), e passo a coordinate radiali con r=r(t) e theta=theta(t); esempio semplice r=t e theta=t, così da avere una curva parametrica nel piano descritta da (t cos t, t sin t). Poi aggiungo la dimensione non informativa z, così da avere (t cos t, t sin t, z) e infine, magari ruoto di due angoli a caso tutta la baracca.
Le dimensioni dello spazio sono 3, quelle della varietà sono 2, ma quella informativa è una sola.
[ritornello 1D]

TRE
Adesso prendo, scrivendo un po’ a caso, (t cos t, t sin t, t^2 cos 3t, t^3 sin 2t, z1, z2, z3), con z1, z2 e z3 non informative. E’ chiaro che la dimensione informativa è una sola. [Ritornello 1D]

QUATTRO
Aggiungo altri due parametri u e v che si combinano tra di loro e con t, in modo che esista f(t,u,v) con tutte e tre le dimensioni informative (esempio f(t,u,v)=parity(ceiling(10 t) ceiling(10 u) ceiling(10 v)))
Adesso combino le cose un po’ a caso
(v sin t cos u, t sin v sin u, v u cos 3 t, t^3 sin 2t, z1, z2, z3), con le z non informative
E’ chiaro che le dimensioni effettive sono 3 [ritornello 3D]

INSOMMA
Basta decidere le dimensionalità k della manifold effettivamente informativa (poi decidere la f() di k variabili), combinare le k variabili con delle funzioni continue e finite ad esempio trigonometriche, e aggiungere delle dimensioni non informative. Poi se uno vuole può anche ruotare nello spazio alto dimensionale. Quel che salta fuori è una roba sufficientemente complessa. E può anche peggiorare facilmente complicando f().


