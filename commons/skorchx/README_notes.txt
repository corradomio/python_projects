Ci sono dei casi in cui bisogna parametrizzare un oggetto
Ma i parametri dipendono dall'oggetto stesso.

skorch usa questa strategia:

    <parameter_name>                        per specificare l'oggetto
    <parameter_name>__<other_parameter>     per parametrizzare l'oggetto

-----------------------------------------------------------------------------

Come specificare i lag:

    singolo parametro contenente le info SIA per y che per X OPPURE
    due parametri?

        lags    == [xlags, ylags]
        tlags

Supponiamo che i lags siano sempre specificati per X e per Y:

        None        [[], []]
        n           [[1,...n], [1...n]]
        [n]         [[1....n], []]
        [n,m]       [[1....n], [...m]]
