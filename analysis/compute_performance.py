import argparse
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from sklearn import metrics
from scipy.stats import ttest_rel

from paths import RUN_DIR
from analysis.process_adversarial_log import process_log_file


experiments_map = {
    'a2e': ['6_7470d_2024-03-21_18-18-51', '21_b0d56_2024-03-21_17-35-21', '42_12d4e_2024-03-21_17-20-52',
            '796_f8e40_2024-03-21_17-49-48', '950_02287_2024-03-21_18-03-59',
            '2025_a54ca_2024-04-08_15-26-33', '3100_c4478_2024-04-08_15-18-00', '2451_d2ddf_2024-04-08_15-09-12',
            '2324_dac90_2024-04-08_15-00-43', '7192_92d91_2024-04-08_14-55-27'],
    'a2b2e': ['6_9b0af_2024-03-23_09-06-57', '21_0a029_2024-03-23_08-45-22', '42_12710_2024-03-23_08-38-33',
              '796_345ee_2024-03-23_08-53-22', '950_8272f_2024-03-23_08-59-14',
              '2025_22be4_2024-04-08_16-05-32', '3100_7095a_2024-04-08_15-57-02', '2451_ce8e5_2024-04-08_15-48-26',
              '2324_26122_2024-04-08_15-42-02', '7192_11a77_2024-04-08_15-33-22'],
    'a2m2e': ['6_ce53b_2024-03-21_18-11-21', '21_71ec2_2024-03-21_17-28-00', '42_141c4_2024-03-21_17-13-21',
              '796_68db8_2024-03-21_17-42-18', '950_450ba_2024-03-21_17-56-30',
              '2025_7378d_2024-04-08_16-50-06', '3100_3454f_2024-04-08_16-41-19', '2451_ca4de_2024-04-08_16-32-15',
              '2324_d8a0a_2024-04-08_16-23-18', '7192_30f3a_2024-04-08_16-14-29'],
    'aa2e': ['6_ae6fe_2024-04-08_03-57-51', '21_cd16a_2024-04-05_16-08-08', '42_01445_2024-04-07_13-12-46',
              '796_f1ba3_2024-04-07_18-04-08', '950_c7521_2024-04-07_23-45-23',
              '2025_b9d49_2024-04-10_12-17-09', '3100_8350a_2024-04-10_07-01-10', '2451_2d3e9_2024-04-10_02-24-03',
              '2324_f02c8_2024-04-09_21-06-54', '7192_69ba9_2024-04-09_16-54-31'],
    'aa2b2e': ['6_85f53_2024-04-08_05-30-04', '21_065d8_2024-04-05_16-07-33', '42_30506_2024-04-07_13-12-23',
               '796_7524d_2024-04-07_17-57-49', '950_e4580_2024-04-07_23-38-55',
               '2025_35cf7_2024-04-10_12-07-54', '3100_46a3a_2024-04-10_08-07-47', '2451_555af_2024-04-10_03-33-40',
               '2324_52036_2024-04-09_22-08-31', '7192_a8c1f_2024-04-09_16-54-50']
}

def opts_parser():
    """ Command line argument setup. """
    parser = argparse.ArgumentParser(description="Script to process adversarial log files.")
    parser.add_argument('--performance_table', action='store_true', default=False, help="Computes performance table (before any attack).")
    parser.add_argument('--statistical_test', action='store_true', default=False,
                        help='Computes statistical significance of A2E vs A2M2E, and A2B2E vs A2M2E.')
    return parser


def get_original_performance():
    """ Computes performance of all systems on clean data, prints table providing overview thereof. """
    adv_path = Path(RUN_DIR) / 'adversaries/'
    mae_dict = {m: None for m in experiments_map.keys()}
    corr_dict = {m: None for m in experiments_map.keys()}
    for model in experiments_map.keys():
        corrs, maes = [], []

        for mn in experiments_map[model]:
            log_file = Path(list(glob(str(adv_path) + '/{}*'.format(mn)))[0]) / 'log.csv'
            orig_metrics, _, _, _, _ = process_log_file(log_file)
            corrs.append(orig_metrics['corr_avg'])
            maes.append(orig_metrics['mae'])

        mae_dict.update({model: [np.array(maes).mean(), np.array(maes).std()]})
        corr_dict.update({model: [np.array(corrs).mean(), np.array(corrs).std()]})

    # create and print table
    table = [['A2E (orig.)', '0.76*', '-'], ['A2M2E (orig.)', '0.75*', '-'],
             ['A2E', '{0:.2f} +/- {1:.2f}'.format(corr_dict['a2e'][0], corr_dict['a2e'][1]),
              '{0:.2f} +/- {1:.2f}'.format(mae_dict['a2e'][0], mae_dict['a2e'][1])],
             ['A2B2E', '{0:.2f} +/- {1:.2f}'.format(corr_dict['a2b2e'][0], corr_dict['a2b2e'][1]),
              '{0:.2f} +/- {1:.2f}'.format(mae_dict['a2b2e'][0], mae_dict['a2b2e'][1])],
             ['A2M2E', '{0:.2f} +/- {1:.2f}'.format(corr_dict['a2m2e'][0], corr_dict['a2m2e'][1]),
              '{0:.2f} +/- {1:.2f}'.format(mae_dict['a2m2e'][0], mae_dict['a2m2e'][1])],
             ['aA2E', '{0:.2f} +/- {1:.2f}'.format(corr_dict['aa2e'][0], corr_dict['aa2e'][1]),
              '{0:.2f} +/- {1:.2f}'.format(mae_dict['aa2e'][0], mae_dict['aa2e'][1])],
             ['aA2B2E', '{0:.2f} +/- {1:.2f}'.format(corr_dict['aa2b2e'][0], corr_dict['aa2b2e'][1]),
              '{0:.2f} +/- {1:.2f}'.format(mae_dict['aa2b2e'][0], mae_dict['aa2b2e'][1])]]
    df = pd.DataFrame(table, columns=['System', 'Avg. Corr', 'MAE'])
    print(df)


def compute_statistical_test():
    """ Computes statistical significance between performance after attack of blackbox vs. interpretable model(s). """
    adv_path = Path(RUN_DIR) / 'adversaries/'
    mae_dict = {m: None for m in experiments_map.keys()}
    for model in experiments_map.keys():
        maes = []

        for mn in experiments_map[model]:
            log_file = Path(list(glob(str(adv_path) + '/{}*'.format(mn)))[0]) / 'log.csv'
            orig_metrics, new_metrics, _, _, _ = process_log_file(log_file)

            true_labels = orig_metrics['true_labels']
            adv_preds = new_metrics['preds']
            maes.extend([metrics.mean_absolute_error(t, a) for t, a in zip(true_labels, adv_preds)])

        maes = np.array(maes)
        mae_dict.update({model: maes})
    statistical_test_pairs = [['a2e', 'a2m2e'], ['a2b2e', 'a2m2e']]

    print('-------------------------------------------------------------------------------------')
    for a, b in statistical_test_pairs:
        print('test pair: {}, {}'.format(a, b))
        res = ttest_rel(mae_dict[a], mae_dict[b])
        print(res)
    print('-------------------------------------------------------------------------------------')


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()

    if opts.performance_table:
        get_original_performance()
    if opts.statistical_test:
        compute_statistical_test()


if __name__ == '__main__':
    main()
