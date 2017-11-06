import cython
import numpy as np
cimport numpy as np
import pandas as pd
ctypedef np.float64_t DTYPE_t



@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.cdivision(True)
def atr_nonull_eqty(eq, int risk_period):
    """
    Wilders ATR (special case which skips zero volatility)
    :return: AverageTrueRange of OHLC
    """
    if risk_period <= 0:
        raise ValueError("period must be positive integer")

    cdef int barcount = len(eq)
    cdef int i = 0


    cdef np.ndarray[DTYPE_t, ndim=1] series = eq.values
    cdef np.ndarray[DTYPE_t, ndim=1] res = np.full(barcount, np.nan)

    cdef float sumtr = 0.0
    cdef float avg = 0.0
    cdef float v = 0.0
    cdef int bucket_cnt = 0
    cdef int period = risk_period

    for i in range(barcount):
        if i == 0:
            continue
        else:
            v = abs(series[i] - series[i-1])

            if bucket_cnt <= period - 1:
                # Skipping points < period
                if v > 0:
                    sumtr += v

                    bucket_cnt += 1

                    # First point is a simple average
                    if bucket_cnt == period - 1:
                        avg = sumtr / period
            else:
                # Wilders smoothing
                if v > 0:
                    avg = ((1.0 / period) * v + (1.0 - 1.0 / period) * avg)
                res[i] = avg
    return pd.Series(res, index=eq.index)