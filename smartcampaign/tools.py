import pandas as pd
import numpy as np
import pyximport; pyximport.install()
from smartcampaign.tools_fast import atr_nonull_eqty


def atr_nonull(H, L, C, period):
    return atr_nonull_eqty(C, period)


def risk_atr(eq, risk_period):
    return atr_nonull_eqty(eq, risk_period)


def risk_atrmax(eq, risk_period):
    atr = atr_nonull_eqty(eq, risk_period)
    return atr.rolling(risk_period).max()


def risk_ddavg(eq, risk_period):
    atr = atr_nonull_eqty(eq, risk_period)
    dd_series = eq - eq.expanding().max()
    avg_dd = dd_series.rolling(risk_period).mean().abs()
    # In case if DD is too low use ATR!
    return np.maximum(atr, avg_dd)


def risk_ddmax(eq, risk_period):
    atr = atr_nonull_eqty(eq, risk_period)
    dd_series = eq - eq.expanding().max()
    dd = dd_series.rolling(risk_period).min().abs()
    # In case if DD is too low use ATR!
    return np.maximum(atr, dd)


def risk_ddq95(eq, risk_period):
    atr = atr_nonull_eqty(eq, risk_period)
    dd_series = eq - eq.expanding().max()
    dd = dd_series.rolling(risk_period).quantile(0.95).abs()
    # In case if DD is too low use ATR!
    return np.maximum(atr, dd)
