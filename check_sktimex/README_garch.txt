https://labdisia.disia.unifi.it/calzolari/materiale-didattico/marchese-1.pdf

prices  P = <P[t], t=1...T>

return  R[t]   = (P[t] - P[t-1])/P[t-1]     = P[t]/P[t-1] - 1
        R[t:1] =

        R[t:k] = (P[t] - P[t-k])/P[t-k]     = P[t]/P[t-k] - 1

        1 + R[t:k] = prod(j=0..k, (1 + R[t-j]))


-----------------------------------------------------------------------------

A quanto sembra *ARCH NON SERVE per le serie temporali!
Analizza la "varianza" NON "i dati"

