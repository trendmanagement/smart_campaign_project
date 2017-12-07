import pandas as pd
from smartcampaign import SmartCampaignBase
from smartcampaign.tools import *
import os

class SmartCampaignCustom_1(SmartCampaignBase):
    def calc_alpha_risk(self, alpha_equity):
        """
        Perform calculation of single alpha risk
        :param alpha_equity: alpha equity series
        :return: risk adjusted alpha equity series
        """
        # Note: CAMPAIGN_DICT values are available via self._cmp_dict

        # Use default method of SmartCampaignBase.calc_alpha_risk()
        return super().calc_alpha_risk(alpha_equity)

    def calc_rel_str(self, df_adj_alpha_equity, group_name=None):
        """
        Calculates relative strength index.

        Algo steps:
        1. Calculate equity index mean of all alphas
        2. Calculate rel str for each alpha: [each alpha] / [equity index]
        3. Calculate the difference of each rel str in 'relstr_mean_per' window
        4. Smooth the #3 using moving average with 'relstr_diff_avg_window' window
        5. Sort values and apply decile weights

        """
        relstr_diff_avg_window = self._cmp_dict['relstr_diff_avg_window']
        relstr_mean_per = self._cmp_dict['relstr_mean_period']
        result_dict = {}

        if group_name in self.tags:

            if group_name == '' or group_name is None:
                eq_long = df_adj_alpha_equity.ffill()
            else:
                eq_long = df_adj_alpha_equity[self.tags[group_name]].ffill()

            if len(eq_long):
                # Calculate all equities index
                eq_mean_idx = eq_long.mean(axis=1)
                rel_str_diff = (eq_long.div(eq_mean_idx, axis=0)).diff(relstr_mean_per).rolling(
                    relstr_diff_avg_window).mean()

                sorted_alphas_rstr = rel_str_diff.iloc[-1].sort_values()

                # Select alphas which are relatively greater than average
                q_weights = self._cmp_dict['relstr_quantiles_weights']

                for i, (alpha_name, rstr_value) in enumerate(sorted_alphas_rstr.items()):
                    #
                    for q, w in q_weights:
                        if i <= int(q * len(sorted_alphas_rstr)):
                            result_dict[alpha_name] = w
                            break

        return result_dict

    def compose_portfolio(self, df_adj_alpha_equity: pd.DataFrame) -> dict:
        """
        Main portfolio composition method (by default it returns all alphas with weight 1.0)
        :param df_adj_alpha_equity: Risk adjusted equity dataframe
        :return: dict[alpha_name: adj_alpha_size]                
        """
        # Note: CAMPAIGN_DICT values are available via self._cmp_dict
        result_dict = {}

        # Calculating relative strength to average LONG-tagged set of equities




        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZS_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZW_groupN1'))
        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZW_groupN2'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'CL_groupN1'))
        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'CL_groupN2'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ES_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6J_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZN_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6B_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6C_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6E_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'HE_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'GC_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'LE_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'NG_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZC_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZL_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'CC_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'SB_groupN1'))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'groupN'))

        return result_dict

    def calc_campaign_risk(self, adj_campaign_equity) -> float:
        """
        Perform estimation of campaign risk
        :param adj_campaign_equity: cumulative campaign equity of composed alphas with adjusted weights
        :return: the estimated risk of campaign composition (float number)
        """
        # Note: CAMPAIGN_DICT values are available via self._cmp_dict

        # Use default method of SmartCampaignBase.calc_campaign_risk()
        return super().calc_campaign_risk(adj_campaign_equity)
