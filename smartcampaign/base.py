import pandas as pd
import numpy as np
from smartcampaign.tools import risk_atr, risk_atrmax, risk_ddavg, risk_ddq95, risk_ddmax, atr_nonull
import math
import re
import warnings
import matplotlib.pyplot as plt
from pydoc import locate
import pickle
import lz4


def object_to_full_path(obj) -> str:
    """
    Converts object instance to full-qualified path like 'package.module.ClassName'
    :param obj: any object class
    :return:
    """
    module_name = obj.__module__

    if module_name == '__main__':
        raise RuntimeError("Serialization of objects from '__main__' namespace is not allowed, if you are using "
                           "Jupyter/IPython session try to save class object to separate module.")

    try:
        # In case when the object is Class (i.e. type itself)
        return "{0}.{1}".format(module_name, obj.__qualname__)
    except AttributeError:
        # The case when the object is class instance
        return "{0}.{1}".format(module_name, obj.__class__.__name__)


def object_from_path(obj_path):
    """
    Loads object class from full-qualified path like 'package.module.ClassName'
    :param obj_path: full-qualified path like 'package.module.ClassName'
    :return:
    """
    obj = locate(obj_path)

    if obj is None:
        raise ValueError("Failed to load object from {0}. Try to check that object path is valid ".format(obj_path) +
                         "and package path in the $PYTHONPATH environment variable")

    return obj

class SmartCampaignBase:
    def __init__(self, campaign_dict, campaign_data_dict, default_commission_per_contr=3.0):
        """
        Initialize base SmartCampaing class
        :param campaign_dict: setting dictionary for Smart Campaign
        :param campaign_data_dict: pandas.DataFrame for alpha equities
        :param default_commission_per_contr: USD commission per contract for trades count estimation
        """
        self._cmp_dict = campaign_dict
        self._cmp_data_dict = campaign_data_dict

        self.default_commission_per_contr = default_commission_per_contr

        # Get products from alphas list
        self.alphas_products = self.get_alphas_product_dict()

        # Convert stacked alphas to single equity series
        self.equities, self.costs, self.exposures, self.deltas, self.ncontracts, self.tags = self._init_dataframes()


    @staticmethod
    def get_alphas_list_from_settings(campaing_dict):
        alphas = []

        for alpha_name, alpha_settings in campaing_dict['alphas'].items():
            if 'alphas' not in alpha_settings:
                # Seems to be single alpha
                alphas.append(alpha_name)
            else:
                # Stacked alpha
                for stacked_alpha in alpha_settings['alphas'].keys():
                    alphas.append(stacked_alpha)

        return alphas

    def get_alphas_product_dict(self):
        alphas = {}

        def _get_product_by_name(alpha_name, product_def='N/A'):
            if alpha_name.startswith("!NEW_"):
                re_match = re.findall(r"_US_(?P<product>[a-zA-Z]*)_", alpha_name)
                if not re_match:
                    if product_def == 'N/A':
                        warnings.warn("Failed to get product from alpha name: {0}".format(alpha_name))
                    product = product_def
                else:
                    product = re_match[0]
            else:
                product = alpha_name.split('_')[0]

            return product

        for alpha_name, alpha_settings in self._cmp_dict['alphas'].items():
            if 'alphas' not in alpha_settings:
                # Seems to be single alpha
                alphas[alpha_name] = _get_product_by_name(alpha_name,
                                                          product_def=alpha_settings.get('product', 'N/A'))
            else:
                # Stacked alpha
                products = set()
                for stacked_alpha in alpha_settings['alphas'].keys():
                    products.add(_get_product_by_name(stacked_alpha,
                                                      product_def=alpha_settings.get('product', 'N/A')))
                if len(products) != 1:
                    raise ValueError("Stacked alpha contains different products! {0}".format(alpha_name))

                alphas[alpha_name] = products.pop()

        return alphas

    def _init_dataframes(self):
        alphas_equity_data = {}
        alphas_costs_data = {}
        alphas_exposure_data = {}
        alphas_delta_data = {}
        alphas_ncontracts_data = {}

        alphas_tags = {}

        for alpha_name, alpha_settings in self._cmp_dict['alphas'].items():
            tags = alphas_tags.setdefault(alpha_settings.get('tag', ''), [])
            tags.append(alpha_name)

            if alpha_name in self._cmp_data_dict:
                # Seems to be non-stacked alpha
                if 'alphas' in alpha_settings:
                    # Oops, we have stacked alpha, but all stacked alphas must have unique names
                    raise KeyError("Alpha name of stacked alpha ({0}) duplicated with single alpha. "
                                   "Make sure that stacked alpha name in unique and doesn't exist in _cmp_dataframe".format(alpha_name))

                alphas_equity_data[alpha_name] = self._cmp_data_dict[alpha_name]['equity']
                alphas_costs_data[alpha_name] = self._cmp_data_dict[alpha_name]['costs']
                alphas_exposure_data[alpha_name] = self._cmp_data_dict[alpha_name]['exposure']
                alphas_delta_data[alpha_name] = self._cmp_data_dict[alpha_name]['delta']
                alphas_ncontracts_data[alpha_name] = self._cmp_data_dict[alpha_name]['costs'].abs() / self.default_commission_per_contr

            elif 'alphas' in alpha_settings:
                # Stacked alpha
                equity_series = None
                costs_series = None
                exposure_series = None
                delta_series = None
                ncontracts_series = None
                for stacked_alpha_name, stacked_alpha_qty in alpha_settings['alphas'].items():

                    if stacked_alpha_name in self._cmp_dict['alphas']:
                        raise KeyError("Duplicated alpha name. {0} used multiple times as single alpha "
                                       "and as stack member".format(stacked_alpha_name))

                    if not isinstance(stacked_alpha_qty, (float, int, np.float64, np.float32, np.int64, np.int32)):
                        # Do type checking, to prevent user wrong inputs
                        raise ValueError("Wrong type of stacked alpha qty for {0}, expected float or int, got {1}".format(
                            alpha_name,
                            type(stacked_alpha_qty)
                        ))

                    if stacked_alpha_name not in self._cmp_data_dict:
                        raise KeyError("Alpha {0} is not found in _cmp_dataframe".format(stacked_alpha_name))
                    if stacked_alpha_qty <= 0:
                        raise ValueError("Qty of stacked alpha leg <= 0. Stacked alpha: {0}".format(alpha_name))

                    if equity_series is None:
                        equity_series = self._cmp_data_dict[stacked_alpha_name]['equity'] * stacked_alpha_qty

                        costs_series = self._cmp_data_dict[stacked_alpha_name]['costs'] * stacked_alpha_qty
                        exposure_series = self._cmp_data_dict[stacked_alpha_name]['exposure'] * stacked_alpha_qty
                        delta_series = self._cmp_data_dict[stacked_alpha_name]['delta'] * stacked_alpha_qty
                        ncontracts_series = self._cmp_data_dict[stacked_alpha_name]['costs'].abs() / self.default_commission_per_contr * stacked_alpha_qty
                    else:
                        equity_series += self._cmp_data_dict[stacked_alpha_name]['equity'] * stacked_alpha_qty
                        costs_series += self._cmp_data_dict[stacked_alpha_name]['costs'] * stacked_alpha_qty
                        exposure_series += self._cmp_data_dict[stacked_alpha_name]['exposure'] * stacked_alpha_qty
                        delta_series += self._cmp_data_dict[stacked_alpha_name]['delta'] * stacked_alpha_qty
                        ncontracts_series += self._cmp_data_dict[stacked_alpha_name]['costs'].abs() / self.default_commission_per_contr * stacked_alpha_qty



                # Final sanity checks
                if equity_series is None:
                    raise ValueError("Stacked alpha is not initiated: {0}".format(alpha_name))

                alphas_equity_data[alpha_name] = equity_series
                alphas_costs_data[alpha_name] = costs_series
                alphas_exposure_data[alpha_name] = exposure_series
                alphas_delta_data[alpha_name] = delta_series
                alphas_ncontracts_data[alpha_name] = ncontracts_series

        return pd.DataFrame(alphas_equity_data), pd.DataFrame(alphas_costs_data), pd.DataFrame(alphas_exposure_data), \
               pd.DataFrame(alphas_delta_data), pd.DataFrame(alphas_ncontracts_data), alphas_tags


    def calc_alpha_risk(self, alpha_equity):
        """
        Perform calculation of single alpha risk
        :param alpha_equity: alpha equity series
        :return: risk adjusted alpha equity series
        """
        if 'alpha_risk_type' not in self._cmp_dict:
            raise KeyError("'alpha_risk_type' is not found in SmartCampaign settings!")

        if 'alpha_risk_period' not in self._cmp_dict:
            raise KeyError("'alpha_risk_period' is not found in SmartCampaign settings!")

        eq = alpha_equity.ffill()

        alpha_risk_type = self._cmp_dict['alpha_risk_type']
        alpha_risk_period = self._cmp_dict['alpha_risk_period']
        alpha_min_risk = self._cmp_dict['alpha_min_risk']

        #
        # Applying alpha risk measure depending on alpha_risk_type
        #
        if alpha_risk_type == 'atr':
            alpha_risk = risk_atr(eq, alpha_risk_period)
        elif alpha_risk_type == 'atrmax':
            alpha_risk = risk_atrmax(eq, alpha_risk_period)
        elif alpha_risk_type == 'ddavg':
            alpha_risk = risk_ddavg(eq, alpha_risk_period)
        elif alpha_risk_type == 'ddmax':
            alpha_risk = risk_ddmax(eq, alpha_risk_period)
        elif alpha_risk_type == 'ddq95':
            alpha_risk = risk_ddq95(eq, alpha_risk_period)
        else:
            raise NotImplementedError('risk_type {0} is not supported!'.format(alpha_risk_type))

        assert alpha_risk.min() > 0, 'Alpha risk must be > 0, got {0}'.format(alpha_risk.min())
        assert alpha_min_risk > 0, 'alpha_min_risk must be > 0, got {0}'.format(alpha_min_risk)
        #
        # ! IMPORTANT: we use $1000 constant to adjust each alpha equity volatility
        #              at the step 2 we will apply campaign based position adjustments
        alpha_size = 1000.0 / np.maximum(alpha_risk, alpha_min_risk)

        # Recalculating equity series based on new adjusted alpha size
        return (alpha_size.shift(1) * eq.diff()).cumsum(), alpha_size

    def compose_portfolio(self, df_adj_alpha_equity: pd.DataFrame) -> dict:
        """
        Main portfolio composition method (by default it returns all alphas with weight 1.0)
        :param df_adj_alpha_equity: Risk adjusted equity dataframe
        :return: dict[alpha_name: adj_alpha_size]
        """

        # by default return all alphas with weight 1.0)
        return {alpha_name: 1.0 for alpha_name in df_adj_alpha_equity}

    def calc_campaign_risk(self, adj_campaign_equity) -> float:
        """
        Perform estimation of campaign risk
        :param adj_campaign_equity: cumulative campaign equity of composed alphas with adjusted weights
        :return: the estimated risk of campaign composition (float number)
        """
        if 'campaign_risk_type' not in self._cmp_dict:
            raise KeyError("'campaign_risk_type' is not found in SmartCampaign settings!")

        if 'campaign_risk_period' not in self._cmp_dict:
            raise KeyError("'campaign_risk_period' is not found in SmartCampaign settings!")

        eq = adj_campaign_equity.ffill()

        campaign_risk_type = self._cmp_dict['campaign_risk_type']
        campaign_risk_period = self._cmp_dict['campaign_risk_period']
        campaign_min_risk = self._cmp_dict['alpha_min_risk']

        #
        # Applying campaign risk measure depending on campaign_risk_type
        #
        if campaign_risk_type == 'atr':
            campaign_risk = risk_atr(eq, campaign_risk_period)
        elif campaign_risk_type == 'atrmax':
            campaign_risk = risk_atrmax(eq, campaign_risk_period)
        elif campaign_risk_type == 'ddavg':
            campaign_risk = risk_ddavg(eq, campaign_risk_period)
        elif campaign_risk_type == 'ddmax':
            campaign_risk = risk_ddmax(eq, campaign_risk_period)
        elif campaign_risk_type == 'ddq95':
            campaign_risk = risk_ddq95(eq, campaign_risk_period)
        else:
            raise NotImplementedError('risk_type {0} is not supported!'.format(campaign_risk_type))

        cmp_risk = campaign_risk[-1]

        assert math.isnan(cmp_risk) or cmp_risk > 0, "Campaign risk must be > 0, got {0}".format(cmp_risk)
        assert campaign_min_risk > 0, "campaign_min_risk must be > 0, got {0}".format(campaign_min_risk)

        return max(cmp_risk, campaign_min_risk)

    def calculate(self, date=None, use_plain_campaign_weights=False):
        """
        Calculate campaign risk and composition at particular date
        :param date:
        :return:
        """

        # Slice alphas
        if date is None:
            # No slicing, use all available data
            alphas_eq = self.equities.ffill()
        else:
            alphas_eq = self.equities.ffill().loc[:date]


        # Calculate every single alpha risks
        adj_alpha_equities_dict = {}
        adj_alpha_size_dict = {}

        for alpha_name in alphas_eq:
            _eq, _size = self.calc_alpha_risk(alphas_eq[alpha_name])
            assert isinstance(_eq, pd.Series), 'self.calc_alpha_risk must return (pd.Series, pd.Series)'
            assert isinstance(_size, pd.Series), 'self.calc_alpha_risk must return (pd.Series, pd.Series)'

            adj_alpha_equities_dict[alpha_name] = _eq
            adj_alpha_size_dict[alpha_name] = _size

        # Compose adjusted equity alpha dataframe
        df_adj_alpha_equities = pd.DataFrame(adj_alpha_equities_dict)
        df_adj_alpha_size = pd.DataFrame(adj_alpha_size_dict)

        # Compose portfolio of alphas and calculate weights
        if use_plain_campaign_weights:
            # Use 1.0 weight for all alphas
            alpha_weights_dict = {c: 1.0 for c in df_adj_alpha_equities.columns}
        else:
            alpha_weights_dict = self.compose_portfolio(df_adj_alpha_equities)

            #
            # Check 'alpha_weights_dict' validity
            #
            if set(alpha_weights_dict.keys()) != set(df_adj_alpha_equities.columns):
                raise KeyError("'self.compose_portfolio' returns different set of alphas")

            if not isinstance(alpha_weights_dict, dict):
                raise ValueError("'self.compose_portfolio' of SmartCampaign must return dict() of {'alpha_name': xx.xx}")
            else:
                for k, v in alpha_weights_dict.items():
                    if not isinstance(k, str):
                        raise ValueError(
                            "'self.compose_portfolio' of SmartCampaign must return dict() of {'alpha_name': xx.xx}")

                    if not isinstance(v, (float, int, np.float64, np.float32, np.int64, np.int32)):
                        raise ValueError(
                            "'self.compose_portfolio' of SmartCampaign must return dict() of {'alpha_name': xx.xx}")

                    if v < 0:
                        raise ValueError("'self.compose_portfolio' of SmartCampaign returned negative alpha weight"
                                         "for alpha {0}".format(k))
                    if math.isnan(v):
                        raise ValueError("'self.compose_portfolio' of SmartCampaign returned NaN alpha weight"
                                         "for alpha {0}".format(k))


        # Calculate total campaign risk
        alpha_weights = pd.Series(alpha_weights_dict)
        total_campaign_equity = (df_adj_alpha_equities * alpha_weights).sum(axis=1)
        campaign_risk = self.calc_campaign_risk(total_campaign_equity)

        if not isinstance(campaign_risk, (float, int, np.float64, np.float32, np.int64, np.int32)):
            raise ValueError("self.calc_campaign_risk must return a number, got {0}".format(type(campaign_risk)))

        if campaign_risk == 0:
            raise ValueError("self.calc_campaign_risk returned zero-campaign risk!")

        # DONE!
        return df_adj_alpha_size.iloc[len(df_adj_alpha_size)-1], alpha_weights, abs(campaign_risk) # campaign_risk must be always positive


    def backtest(self, **kwargs):
        """
        Backtest SmartCampaign
        **kwargs:
        - 'initial_capital' - 100000 by default
        - 'target_risk_percent' - 0.1 by default (i.e. 10%)
        - 'start_date' - None by default (as data availability)
        :return:
        """

        # ! Starting date doesn't shrink the data for risk calculation! We need 2 dates!

        # Init equities and capital
        initial_capital = kwargs.get('initial_capital', 100000)  # By default 100 000
        target_risk_percent = kwargs.get('target_risk_percent', 0.1)  # By default 0.1 (i.e. 10%)
        start_date = kwargs.get('start_date', None) # By default None (use all data as possible)

        # Equity line of Smart campaign
        equity_mm = pd.Series(initial_capital, index=self.equities.index, dtype=np.float64)
        # Equity line of risk adjusted alphas but without alpha selection / rebalancing
        equity_plain_adj = pd.Series(initial_capital, index=self.equities.index, dtype=np.float64)
        # Equity line of original alphas
        equity_plain_adj_noreinv = pd.Series(initial_capital, index=self.equities.index, dtype=np.float64)
        # Campaign Series
        campaign_risk_series = pd.Series(float('nan'), index=self.equities.index, dtype=np.float64)
        campaign_plain_risk_series = pd.Series(float('nan'), index=self.equities.index, dtype=np.float64)

        campaign_alphas_size = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)
        campaign_alphas_size_raw = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns,
                                            dtype=np.float64)
        alphas_eq_plain = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)
        alphas_eq_mm = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)

        alphas_costs_mm = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)
        alphas_delta_mm = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)
        alphas_nexecuted_mm = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)

        # Main campaign settings
        alpha_adj_weights = None
        alpha_cmp_weights = None
        cmp_total_weights = None
        cmp_risk = None
        cmp_size = None
        prev_date = None
        plain_cmp_total_weights = None

        source_eq_diff = self.equities.diff()

        # For each day calculate the equity
        for i, dt in enumerate(self.equities.index):
            if i == 0:
                prev_date = dt
                continue

            if start_date is not None:
                if dt.date() < start_date.date():
                    continue

            # Apply current alpha weights to PnL calculation
            if cmp_risk is not None and math.isfinite(cmp_risk) and equity_mm[i-1] > 0:  # prev_equity < 0 margin call!
                base_diff_at_dt = source_eq_diff.iloc[i].fillna(0)
                equity_mm[i] = equity_mm[i-1] + (base_diff_at_dt * cmp_total_weights).sum()
                equity_plain_adj[i] = equity_plain_adj[i-1] + (base_diff_at_dt * plain_cmp_total_weights).sum()
                equity_plain_adj_noreinv[i] = equity_plain_adj_noreinv[i-1] + (base_diff_at_dt * plain_noreinv_cmp_total_weights).sum()

                # Build individual alpha equity
                # Build cumulative alpha equity
                # Build alpha size (rounded) array

                campaign_risk_series[i] = cmp_risk
                campaign_plain_risk_series[i] = plain_cmp_risk
                campaign_alphas_size.loc[dt] = cmp_total_weights
                campaign_alphas_size_raw.loc[dt] = cmp_total_weights_raw

                # calculate individual alpha equities
                alphas_eq_mm.iloc[i] = alphas_eq_mm.iloc[i-1] + base_diff_at_dt * cmp_total_weights
                alphas_eq_plain.iloc[i] = alphas_eq_plain.iloc[i-1] + base_diff_at_dt * plain_cmp_total_weights

                # Calculate deltas
                alphas_costs_mm.iloc[i] = self.costs.iloc[i] * cmp_total_weights
                alphas_delta_mm.iloc[i] = self.deltas.iloc[i] * cmp_total_weights
                alphas_nexecuted_mm.iloc[i] = self.ncontracts.iloc[i] * cmp_total_weights
            else:
                equity_mm[i] = equity_mm[i-1]
                equity_plain_adj[i] = equity_plain_adj[i-1]
                equity_plain_adj_noreinv[i] = equity_plain_adj_noreinv[i-1]
                alphas_eq_mm.iloc[i] = alphas_eq_mm.iloc[i - 1]
                alphas_eq_plain.iloc[i] = alphas_eq_plain.iloc[i - 1]

            # IMPORTANT: this section must be after equity calculation,
            #            because Monday's PnL is the result of Friday's position size

            # If the last day was on previous week (weekend)
            if dt.dayofweek < 2 and prev_date.dayofweek >= 3:
                # We have new week !
                # 1. Run self.calculate(date=Sunday)
                alpha_adj_weights, alpha_cmp_weights, cmp_risk = self.calculate(date=prev_date)
                # And calculate plain weights for comparison
                plain_alpha_adj_weights, plain_alpha_cmp_weights, plain_cmp_risk = self.calculate(date=prev_date,
                                                                                                  use_plain_campaign_weights=True)

                # 2. Apply new alpha weights and campaign risk value
                cmp_size = (equity_mm[i-1] * target_risk_percent) / cmp_risk
                plain_cmp_size = (equity_plain_adj[i-1] * target_risk_percent) / plain_cmp_risk
                noreinv_cmp_size = (initial_capital * target_risk_percent) / plain_cmp_risk

                # Do alpha weights rounding and total campaign size adjustments
                cmp_total_weights = (alpha_adj_weights * alpha_cmp_weights * cmp_size).round()
                cmp_total_weights_raw = (alpha_adj_weights * alpha_cmp_weights)
                plain_cmp_total_weights = (plain_alpha_adj_weights * plain_cmp_size).round()
                plain_noreinv_cmp_total_weights = (plain_alpha_adj_weights * noreinv_cmp_size).round()



            prev_date = dt

        # Return results
        sdate = self.equities.index[0]
        if start_date is not None:
            sdate = start_date
        return {
            'initial_capital': initial_capital,
            'target_risk_percent': target_risk_percent,
            'equity_mm': equity_mm.loc[sdate:],
            'equity_plain_adj': equity_plain_adj.loc[sdate:],
            'equity_plain_adj_noreinv': equity_plain_adj_noreinv.loc[sdate:],
            'equity_simple_sum': self.equities.ffill().loc[sdate:].diff().sum(axis=1).cumsum() + initial_capital,
            'campaign_estimated_base_risk': campaign_risk_series.loc[sdate:],
            'campaign_estimated_base_risk_plain': campaign_plain_risk_series.loc[sdate:],
            'campaign_alphas_size': campaign_alphas_size.loc[sdate:],
            'campaign_alphas_size_raw': campaign_alphas_size_raw.loc[sdate:],
            'alphas_equity_plain': alphas_eq_plain.loc[sdate:],
            'alphas_equity_mm': alphas_eq_mm.loc[sdate:],

            'alphas_deltas_mm': alphas_delta_mm.loc[sdate:],
            'alphas_costs_mm': alphas_costs_mm.loc[sdate:],
            'alphas_contracts_mm': alphas_nexecuted_mm.loc[sdate:],
        }

    def calc_stats(self, eq, initial_capital):
        mm_eq = eq.ffill()
        mm_netprofit = mm_eq[-1]
        mm_atr_series = atr_nonull(mm_eq, mm_eq, mm_eq, self._cmp_dict['campaign_risk_period'])
        mm_max_dd_series = mm_eq - mm_eq.expanding().max()
        mm_max_dd_percent_series = (mm_max_dd_series / mm_eq.expanding().max())  * 100
        return {
            'net_profit_usd': mm_netprofit - initial_capital,
            'net_profit_pct': (mm_netprofit - initial_capital) / initial_capital * 100,

            'mdd_usd': mm_max_dd_series.min(),
            'mdd_pct': mm_max_dd_percent_series.min(),

            'mdd_usd_series': mm_max_dd_series,
            'mdd_pct_series': mm_max_dd_percent_series,

            'atr_max_usd': mm_atr_series.max(),
            'atr_q95_usd': mm_atr_series.dropna().quantile(0.95),
            'atr_avg_usd': mm_atr_series.mean(),

            'atr_max_pct': (mm_atr_series / mm_eq).max() * 100,
            'atr_q95_pct': (mm_atr_series / mm_eq).dropna().quantile(0.95) * 100,
            'atr_avg_pct': (mm_atr_series / mm_eq).mean() * 100,
        }

    def report_money_management(self, bt_stats_dict, alt_dollar_costs=None, performance_fee=0.2, fixed_mgmt_fee=0.02, plot_graph=False):
            eq = bt_stats_dict['equity_mm'].fillna(0.0)
            costs_sum = bt_stats_dict['alphas_costs_mm'].sum(axis=1).cumsum()
            equity_without_costs = (eq - costs_sum)

            #
            # Calculating equity with new costs
            #
            ncontracts_traded = bt_stats_dict['alphas_contracts_mm'].sum(axis=1)
            new_costs = ncontracts_traded * -abs(alt_dollar_costs if alt_dollar_costs is not None else self.default_commission_per_contr)
            new_equity = equity_without_costs + new_costs.cumsum() + bt_stats_dict['initial_capital']

            #
            # Calculation of the performance fees (with high-water mark)
            #
            monthly_eq = new_equity.resample('M').last()
            monthly_high_watermark = monthly_eq.expanding().max().shift()

            # Skip periods when equity closed lower than previous month's high-water mark
            performance_fee_base = monthly_eq - monthly_high_watermark
            performance_fee_base[performance_fee_base <= 0] = 0
            performance_fee = performance_fee_base * -abs(performance_fee)

            management_fee = pd.Series(-abs(fixed_mgmt_fee), index=performance_fee.index)

            performance_fees_sum = performance_fee.cumsum().reindex(eq.index, method='ffill')
            management_fee_sum = management_fee.cumsum().reindex(eq.index, method='ffill')
            performance_fee_equity = new_equity + performance_fees_sum.fillna(0.0) + management_fee_sum.fillna(0.0)

            df_result = pd.DataFrame({
                "equity_original": eq + bt_stats_dict['initial_capital'],
                "equity_with_costs": new_equity,
                "equity_all_included": performance_fee_equity,
                "costs_sum": new_costs.cumsum(),
                'performance_fee_sum': performance_fees_sum,
                'management_fee_sum': management_fee_sum,
                'ncontracts_traded': ncontracts_traded,
                'costs': new_costs,
            })

            if plot_graph:
                df_result[["equity_original", "equity_with_costs", "equity_all_included"]].plot()

                plt.figure()
                df_result[["costs_sum", 'performance_fee_sum', 'management_fee_sum']].plot()
            return df_result

    def report_execution_info(self, bt_stats_dict):
        """
        Generate information for execution (deltas, costs, contracts executed, etc)
        :param bt_stats_dict:
        :return:
        """

        products_list = set(self.alphas_products.values())

        #
        # Deltas by product
        #
        _deltas = bt_stats_dict['alphas_deltas_mm']
        _deltas_df = pd.DataFrame(0, index=_deltas.index, columns=products_list)

        for alpha_name in _deltas.columns:
            _prod = self.alphas_products[alpha_name]
            _deltas_df[_prod] += _deltas[alpha_name]

        plt.figure();
        _deltas_df.plot();
        plt.legend(loc=2);
        plt.title("Deltas by product");

        #
        # Costs by product
        #
        _costs = bt_stats_dict['alphas_costs_mm']
        _costs_df = pd.DataFrame(0, index=_costs.index, columns=products_list)

        for alpha_name in _costs.columns:
            _prod = self.alphas_products[alpha_name]
            _costs_df[_prod] += _costs[alpha_name]

        plt.figure();
        _costs_df.plot();
        plt.legend(loc=2);
        plt.title("Costs by product");

        #
        # Contracts executed
        #
        _ncontracts = bt_stats_dict['alphas_contracts_mm']
        _ncontracts_df = pd.DataFrame(0, index=_ncontracts.index, columns=products_list)

        for alpha_name in _ncontracts.columns:
            _prod = self.alphas_products[alpha_name]
            _ncontracts_df[_prod] += _ncontracts[alpha_name]

        plt.figure();
        _ncontracts_df.plot();
        plt.legend(loc=2);
        plt.title("Number contracts executed by product");





    def report(self, bt_stats_dict):
        """
        Generate backtesting report
        :param bt_stats_dict:
        :return:
        """



        initial_capital = bt_stats_dict['initial_capital']

        def print_multi(field_name, key, dict_list, _format="{0:>10}"):
            pattern = "{0:<30}"
            values = []
            for i, d in enumerate(dict_list):
                pattern += _format.replace("0:", "{0}:".format(i+1))

                if key is None:
                    values.append(d)
                else:
                    values.append(d[key])

            print(pattern.format(field_name, *values))

        stats_mm = self.calc_stats(bt_stats_dict['equity_mm'], initial_capital)
        stats_plain_adj = self.calc_stats(bt_stats_dict['equity_plain_adj'], initial_capital)
        stats_plain_adj_noreinv = self.calc_stats(bt_stats_dict['equity_plain_adj_noreinv'], initial_capital)
        stats_simple_sum = self.calc_stats(bt_stats_dict['equity_simple_sum'], initial_capital)

        eq_list = [stats_mm, stats_plain_adj, stats_plain_adj_noreinv, stats_simple_sum]

        print_multi('', None, ['   MM', '   Adj Plain', "   Adj No Reinv", "   Simple Sum"], _format="{0:<15}")

        fmt_float = '{0:>15.2f}'
        fmt_float_pct = '{0:>14.2f}%'
        print_multi("NetProfit $", 'net_profit_usd', eq_list, _format=fmt_float)
        print_multi("NetProfit %", 'net_profit_pct', eq_list, _format=fmt_float_pct)
        print('') # Line break

        print_multi("MaxDD $", 'mdd_usd', eq_list, _format=fmt_float)
        print_multi("MaxDD %", 'mdd_pct', eq_list, _format=fmt_float_pct)
        print('')  # Line break

        print_multi("MaxATR $",   'atr_max_usd', eq_list, _format=fmt_float)
        print_multi("Q95% ATR $", 'atr_q95_usd', eq_list, _format=fmt_float)
        print_multi("Avg ATR $",  'atr_avg_usd', eq_list, _format=fmt_float)
        print('')  # Line break

        print_multi("MaxATR %",   'atr_max_pct', eq_list, _format=fmt_float_pct)
        print_multi("Q95% ATR %", 'atr_q95_pct', eq_list, _format=fmt_float_pct)
        print_multi("Avg ATR %",  'atr_avg_pct', eq_list, _format=fmt_float_pct)
        print('')  # Line break



        #
        # Plotting
        #

        # Equities
        plt.figure();
        bt_stats_dict['equity_mm'].plot(label='Eqty MM');
        bt_stats_dict['equity_plain_adj'].plot(label='Eqty Adj Plain');
        bt_stats_dict['equity_plain_adj_noreinv'].plot(label='Eqty Adj No Reinv');
        bt_stats_dict['equity_simple_sum'].plot(label='Eqty Simple Sum');

        plt.legend(loc=2);
        plt.title("Equities");

        # Max DD Series
        plt.figure();
        stats_mm['mdd_pct_series'].plot(label="MaxDD % MM")
        stats_plain_adj['mdd_pct_series'].plot(label="MaxDD % Adj Plain")
        stats_plain_adj_noreinv['mdd_pct_series'].plot(label="MaxDD % Adj No Reinv")
        stats_simple_sum['mdd_pct_series'].plot(label="MaxDD % Simple Sum")

        plt.legend(loc=2);
        plt.title("Max DD percent");

        # Alpha equities
        plt.figure();
        bt_stats_dict['alphas_equity_mm'].plot().legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                                                          loc="lower left",
                                                          mode="expand",
                                                          borderaxespad=0, ncol=1);

        plt.title('Individual alpha equities (MM)', y=0.95);

        plt.figure();
        bt_stats_dict['alphas_equity_plain'].plot().legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                                                        loc="lower left",
                                                        mode="expand",
                                                        borderaxespad=0, ncol=1);

        plt.title('Individual alpha equities (plain)', y=0.95);

        # Alphas sizes
        plt.figure();
        bt_stats_dict['campaign_alphas_size'].plot().legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                                                            loc="lower left",
                                                            mode="expand",
                                                            borderaxespad=0, ncol=1);

        plt.title('Alpha size', y=0.95);

        # Alphas sizes RAW
        plt.figure();
        bt_stats_dict['campaign_alphas_size_raw'].plot().legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                                                            loc="lower left",
                                                            mode="expand",
                                                            borderaxespad=0, ncol=1);

        plt.title('Raw alphas size (without rounding and account risk adj)', y=0.95);


        # Campaign estimated base risk
        plt.figure();
        bt_stats_dict['campaign_estimated_base_risk'].plot(label='Risk MM');
        bt_stats_dict['campaign_estimated_base_risk_plain'].plot(label='Risk Plain');
        plt.legend(loc=2);
        plt.title("Campaign estimated base risk");

        plt.figure();
        bt_stats_dict['alphas_costs_mm'].sum(axis=1).cumsum().plot();
        plt.title("Cumulative costs for all alphas (MM)");

    def save(self, mongo_collection):
        self._cmp_dict['class'] = object_to_full_path(self)

        # Dumping Smartcampaign settings to the collection
        db_collection = mongo_collection
        db_collection.replace_one({'name': self.name}, self._cmp_dict, upsert=True)

    @staticmethod
    def load_from_v1(smart_campaign_name, exo_storage, mongo_collection):
        db_collection = mongo_collection
        smart_dict = db_collection.find_one({'name': smart_campaign_name})

        if smart_dict is None:
            raise KeyError("SmartCampaign {0} is not found in the MongoDB".format(smart_campaign_name))

        SmartCampaignClass = object_from_path(smart_dict['class'])

        # Loading V1 and V2 alphas
        alphas_list = SmartCampaignBase.get_alphas_list_from_settings(smart_dict)
        alphas_series_dict = exo_storage.swarms_data(alphas_list, load_v2_alphas=True)
        df_alphas_equities = pd.DataFrame({k: v['swarm_series']['equity'] for k, v in alphas_series_dict.items()})

        return SmartCampaignClass(smart_dict, df_alphas_equities)

    def export_to_v1_campaign(self):
        """
        Exports to v1 campaign settings
        :return:
        """

        alpha_adj_weights, alpha_cmp_weights, cmp_risk = self.calculate(date=None)

        alpha_total_weights = alpha_adj_weights * alpha_cmp_weights

        campaign_dict = {
            'name': self.name,
            'description': self.description,
            'type': 'smart',
            'alphas': {k: {'qty': v} for k, v in alpha_total_weights.items()},
            'campaign_risk': cmp_risk,
        }

        return campaign_dict

    @property
    def description(self):
        return self._cmp_dict.get('description', '')

    @property
    def name(self):
        return self._cmp_dict['name']

