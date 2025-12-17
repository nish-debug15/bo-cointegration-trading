import numpy as np

def trade(z, entry, exit):
    position = 0
    pnl = np.zeros(len(z))

    for t in range(1, len(z)):
        if position == 0:
            if z[t-1] > entry:
                position = -1
            elif z[t-1] < -entry:
                position = 1
        elif position == 1 and z[t-1] > -exit:
            position = 0
        elif position == -1 and z[t-1] < exit:
            position = 0

        pnl[t] = position * (z[t] - z[t-1])

    return pnl
