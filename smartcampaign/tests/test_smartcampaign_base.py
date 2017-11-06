import unittest
import pandas as pd
import numpy as np
from smartcampaign.base import SmartCampaignBase


class SmartCampaingTestCase(unittest.TestCase):
    def setUp(self):
        # Init sample alpha data
        alpha1 = pd.Series([1, 2, 3, 4, 5, 6])
        alpha3 = pd.Series([-1, -2, -3, -4, -5, -6])
        alpha2 = pd.Series([10, 11, 12, 13, 14, 15])

        self.alpha_df = pd.DataFrame({'alpha1': alpha1,
                                      'alpha2': alpha2,
                                      'alpha3': alpha3,
                                      })

        self.smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {'tag': '',
                                  'alphas': {
                                      'alpha1': 1.0,
                                      'alpha3': 2.0,
                                  }
                                  }
            }
        }

    def test_init(self):
        cmp = SmartCampaignBase(self.smartcampaign_dict, self.alpha_df)
        self.assertEqual(cmp.name, "SmartCampaignTest")

    def test_dataframe_init(self):
        cmp = SmartCampaignBase(self.smartcampaign_dict, self.alpha_df)

        self.assertEqual(pd.DataFrame, type(cmp.equities))

        self.assertTrue('alpha2' in cmp.equities)
        self.assertTrue('alpha1' not in cmp.equities)
        self.assertTrue('alpha1' not in cmp.equities)
        self.assertTrue('alpha_stacked' in cmp.equities)

        self.assertTrue(np.all(cmp.equities['alpha2'] == self.alpha_df['alpha2']))

        self.assertTrue(np.all(cmp.equities['alpha_stacked'] == (self.alpha_df['alpha1']*1.0+self.alpha_df['alpha3']*2.0)))

    def test_dataframe_init_errors_no_errors(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha1': 1.0,
                        'alpha3': 2.0,
                    }
                }
            }
        }
        SmartCampaignBase(smartcampaign_dict, self.alpha_df)

    def test_dataframe_init_errors_alpha_in_df_but_stacked(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha1': {
                    'tag': '',
                    'alphas': {
                        'alpha1': 1.0,
                        'alpha3': 2.0,
                    }
                }
            }
        }
        self.assertRaises(KeyError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)

    def test_dataframe_init_errors_missing_stacked_leg(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha_missing': 1.0,
                        'alpha3': 2.0,
                    }
                }
            }
        }
        self.assertRaises(KeyError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)

    def test_dataframe_init_errors_missing_stacked_leg_negative(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha1': 0.0,
                        'alpha3': 2.0,
                    }
                }
            }
        }
        self.assertRaises(ValueError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)

        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha1': -1.0,
                        'alpha3': 2.0,
                    }
                }
            }
        }
        self.assertRaises(ValueError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)

    def test_dataframe_init_errors_alpha_name_duplicate(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha1': 1.0,
                        'alpha2': 2.0,
                    }
                }
            }
        }
        self.assertRaises(KeyError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)

    def test_dataframe_init_errors_stacked_not_initiated(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {}
                }
            }
        }
        self.assertRaises(ValueError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)

    def test_dataframe_init_errors_stacked_qty_wrong_type(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha1': 'string',
                        'alpha3': 2.0,
                    }
                }
            }
        }
        self.assertRaises(ValueError, SmartCampaignBase, smartcampaign_dict, self.alpha_df)


    def test_alphas_tags(self):
        smartcampaign_dict = {
            'name': "SmartCampaignTest",
            'alphas': {
                'alpha2': {'tag': ''},
                'alpha_stacked': {
                    'tag': '',
                    'alphas': {
                        'alpha1': 1.0,
                        'alpha3': 2.0,
                    }
                }
            }
        }
        cmp = SmartCampaignBase(smartcampaign_dict, self.alpha_df)

        self.assertEqual(dict, type(cmp.tags))
        self.assertEqual(1, len(cmp.tags))
        self.assertEqual(['alpha2', 'alpha_stacked'], list(sorted(cmp.tags[''])))

    def test_get_alphas_list_from_settings(self):

        alphas = SmartCampaignBase.get_alphas_list_from_settings(self.smartcampaign_dict)
        self.assertEqual(list(sorted(alphas)), list(sorted(['alpha2', 'alpha1', 'alpha3'])))





if __name__ == '__main__':
    unittest.main()
