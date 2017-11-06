import pandas as pd
import numpy as np
from smartcampaign.tools import risk_atr, risk_atrmax, risk_ddavg, risk_ddq95, risk_ddmax, atr_nonull
import math
import matplotlib.pyplot as plt


class SmartCampaignBase:
    def __init__(self, campaign_dict, campaign_dataframe):
        """
        Initialize base SmartCampaing class
        :param campaign_dict: setting dictionary for Smart Campaign
        :param campaign_dataframe: pandas.DataFrame for alpha equities
        """
        self._cmp_dict = campaign_dict
        self._cmp_dataframe = campaign_dataframe

        # Convert stacked alphas to single equity series
        self.equities, self.tags = self._init_dataframes()

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

    def _init_dataframes(self):
        alphas_df_data = {}
        alphas_tags = {}

        for alpha_name, alpha_settings in self._cmp_dict['alphas'].items():
            tags = alphas_tags.setdefault(alpha_settings.get('tag', ''), [])
            tags.append(alpha_name)

            if alpha_name in self._cmp_dataframe:
                # Seems to be non-stacked alpha
                if 'alphas' in alpha_settings:
                    # Oops, we have stacked alpha, but all stacked alphas must have unique names
                    raise KeyError("Alpha name of stacked alpha ({0}) duplicated with single alpha. "
                                   "Make sure that stacked alpha name in unique and doesn't exist in _cmp_dataframe".format(alpha_name))

                alphas_df_data[alpha_name] = self._cmp_dataframe[alpha_name]
            elif 'alphas' in alpha_settings:
                # Stacked alpha
                series = None
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

                    if stacked_alpha_name not in self._cmp_dataframe:
                        raise KeyError("Alpha {0} is not found in _cmp_dataframe".format(stacked_alpha_name))
                    if stacked_alpha_qty <= 0:
                        raise ValueError("Qty of stacked alpha leg <= 0. Stacked alpha: {0}".format(alpha_name))

                    if series is None:
                        series = self._cmp_dataframe[stacked_alpha_name] * stacked_alpha_qty
                    else:
                        series += self._cmp_dataframe[stacked_alpha_name] * stacked_alpha_qty

                # Final sanity checks
                if series is None:
                    raise ValueError("Stacked alpha is not initiated: {0}".format(alpha_name))

                alphas_df_data[alpha_name] = series

        return pd.DataFrame(alphas_df_data), alphas_tags

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
        alphas_eq_plain = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)
        alphas_eq_mm = pd.DataFrame(0, index=self.equities.index, columns=self.equities.columns, dtype=np.float64)

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

                # calculate individual alpha equities
                alphas_eq_mm.iloc[i] = alphas_eq_mm.iloc[i-1] + base_diff_at_dt * cmp_total_weights
                alphas_eq_plain.iloc[i] = alphas_eq_plain.iloc[i-1] + base_diff_at_dt * plain_cmp_total_weights
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
            'campaign_estimated_base_risk': campaign_risk_series.loc[sdate:],
            'campaign_estimated_base_risk_plain': campaign_plain_risk_series.loc[sdate:],
            'campaign_alphas_size': campaign_alphas_size.loc[sdate:],
            'alphas_equity_plain': alphas_eq_plain.loc[sdate:],
            'alphas_equity_mm': alphas_eq_mm.loc[sdate:],
        }

    def report(self, bt_stats_dict):
        """
        Generate backtesting report
        :param bt_stats_dict:
        :return:
        """
        initial_capital = bt_stats_dict['initial_capital']
        target_risk_percent = bt_stats_dict['target_risk_percent']

        eq = bt_stats_dict['equity_mm'].ffill()
        netprofit = eq[-1]
        atr_series = atr_nonull(eq, eq, eq, self._cmp_dict['campaign_risk_period'])

        max_dd_series = eq - eq.expanding().max()

        print("{0:<30}{1:>10.2f}".format("NetProfit $", netprofit - initial_capital))
        print("{0:<30}{1:>10.2f}%".format("NetProfit %", (netprofit - initial_capital) / initial_capital * 100))
        print('') # Line break

        print("{0:<30}{1:>10.2f}".format("MaxDD $", max_dd_series.min()))
        print("{0:<30}{1:>10.2f}%".format("MaxDD %", (max_dd_series / eq).min() * 100))
        print('')  # Line break

        print("{0:<30}{1:>10.2f}".format("MaxATR $", atr_series.max()))
        print("{0:<30}{1:>10.2f}".format("Q95% ATR $", atr_series.dropna().quantile(0.95)))
        print("{0:<30}{1:>10.2f}".format("Avg ATR $", atr_series.mean()))
        print('')  # Line break

        print("{0:<30}{1:>10.2f}%".format("MaxATR %", (atr_series/eq).max()*100))
        print("{0:<30}{1:>10.2f}%".format("Q95% ATR %", (atr_series/eq).dropna().quantile(0.95)*100))
        print("{0:<30}{1:>10.2f}%".format("Avg ATR %", (atr_series/eq).mean()*100))

        #
        # Plotting
        #

        # Equities
        plt.figure();
        bt_stats_dict['equity_mm'].plot(label='Eqty MM');
        bt_stats_dict['equity_plain_adj'].plot(label='Eqty Adj Plain');
        bt_stats_dict['equity_plain_adj_noreinv'].plot(label='Eqty Adj No Reinv');
        plt.legend(loc=2);
        plt.title("Equities");

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


        # Campaign estimated base risk
        plt.figure();
        bt_stats_dict['campaign_estimated_base_risk'].plot(label='Risk MM');
        bt_stats_dict['campaign_estimated_base_risk_plain'].plot(label='Risk Plain');
        plt.legend(loc=2);
        plt.title("Campaign estimated base risk");











    @property
    def name(self):
        return self._cmp_dict['name']

