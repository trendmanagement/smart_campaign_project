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

    def calc_rel_str(self, df_adj_alpha_equity, group_name, relstr_mean_per):
        result_dict = {}

        if group_name in self.tags:

            eq_long = df_adj_alpha_equity[self.tags[group_name]].ffill()

            if len(eq_long):
                # Calculate all equities index
                eq_chg = eq_long.diff(relstr_mean_per)
                eq_mean = eq_chg.mean(axis=1)

                # Select alphas which are relatively greater than average
                for alpha_name in eq_chg:
                    if eq_chg[alpha_name][-1] >= eq_mean[-1]:
                        result_dict[alpha_name] = 0.2
                    else:
                        result_dict[alpha_name] = 0.7

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
        relstr_mean_per = self._cmp_dict['relstr_mean_period']

        #         result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'long', relstr_mean_per))
        #         result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'short', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZS_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZW_groupN1', relstr_mean_per))
        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZW_groupN2', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'CL_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ES_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6J_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZN_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6B_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6C_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, '6E_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'HE_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'GC_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'LE_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'NG_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZC_groupN1', relstr_mean_per))

        result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'ZL_groupN1', relstr_mean_per))

        #         result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'groupN1', relstr_mean_per))
        #         result_dict.update(self.calc_rel_str(df_adj_alpha_equity, 'groupN2', relstr_mean_per))


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