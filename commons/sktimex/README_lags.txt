For compatibility with sktime we use:

    xylags:     lags for the past data
                0 is the index of the last slot. It is also the 'cutoff'
    tlags:      lags for the future/prediction data
                1 is the index of the first slot to predict. It is 'cutoff+1'


                    y
    tlags:  n       [1..n]
            [n]     [n]

                              y        X
    xylags  n               [[0..n-1],[0..n-1]]
            [n]             [n,0]
            [n,0]           [[0..n-1],[]]
            [0,m]           [[],[0..m-1]]
            [n,m]           [[0..n-1],[0..m-1]]
            [[..],[..]]     as is


Extension using a dictionary

    {
        <factor>: n         <factor>*[0...(n-1)]
        <factor>:[i, n]     <factor>*[i...(i+n-1)]
    }
