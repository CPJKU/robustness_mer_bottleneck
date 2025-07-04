import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from glob import glob
from pathlib import Path
from paths import RUN_DIR, HOME_DIR
from analysis.process_adversarial_log import process_log_file, convert_strings_to_array


experiments_map = {
    '6_9b0af_2024-03-23_09-06-57': 'A2B2E',
    '6_7470d_2024-03-21_18-18-51': 'A2E',
    '6_ce53b_2024-03-21_18-11-21': 'A2M2E',
    '6_85f53_2024-04-08_05-30-0': 'aA2B2E',
    '6_ae6fe_2024-04-08_03-57-51': 'aA2E',
    '21_0a029_2024-03-23_08-45-22': 'A2B2E',
    '21_71ec2_2024-03-21_17-28-00': 'A2M2E',
    '21_b0d56_2024-03-21_17-35-21': 'A2E',
    '21_065d8_2024-04-05_16-07-33': 'aA2B2E',
    '21_cd16a_2024-04-05_16-08-08': 'aA2E',
    '42_12d4e_2024-03-21_17-20-52': 'A2E',
    '42_141c4_2024-03-21_17-13-21': 'A2M2E',
    '42_12710_2024-03-23_08-38-33': 'A2B2E',
    '42_30506_2024-04-07_13-12-23': 'aA2B2E',
    '42_01445_2024-04-07_13-12-46': 'aA2E',
    '796_68db8_2024-03-21_17-42-18': 'A2M2E',
    '796_345ee_2024-03-23_08-53-22': 'A2B2E',
    '796_f8e40_2024-03-21_17-49-48': 'A2E',
    '796_7524d_2024-04-07_17-57-49': 'aA2B2E',
    '796_f1ba3_2024-04-07_18-04-08': 'aA2E',
    '950_450ba_2024-03-21_17-56-30': 'A2M2E',
    '950_02287_2024-03-21_18-03-59': 'A2E',
    '950_8272f_2024-03-23_08-59-14': 'A2B2E',
    '950_e4580_2024-04-07_23-38-55': 'aA2B2E',
    '950_c7521_2024-04-07_23-45-23': 'aA2E',
    '2025_a54ca_2024-04-08_15-26-33': 'A2E',
    '2025_22be4_2024-04-08_16-05-32': 'A2B2E',
    '2025_7378d_2024-04-08_16-50-06': 'A2M2E',
    '2025_35cf7_2024-04-10_12-07-54': 'aA2B2E',
    '2025_b9d49_2024-04-10_12-17-09': 'aA2E',
    '2324_dac90_2024-04-08_15-00-43': 'A2E',
    '2324_26122_2024-04-08_15-42-02': 'A2B2E',
    '2324_d8a0a_2024-04-08_16-23-18': 'A2M2E',
    '2324_52036_2024-04-09_22-08-31': 'aA2B2E',
    '2324_f02c8_2024-04-09_21-06-54': 'aA2E',
    '2451_d2ddf_2024-04-08_15-09-12': 'A2E',
    '2451_ce8e5_2024-04-08_15-48-26': 'A2B2E',
    '2451_ca4de_2024-04-08_16-32-15': 'A2M2E',
    '2451_555af_2024-04-10_03-33-40': 'aA2B2E',
    '2451_2d3e9_2024-04-10_02-24-03': 'aA2E',
    '3100_c4478_2024-04-08_15-18-00': 'A2E',
    '3100_7095a_2024-04-08_15-57-02': 'A2B2E',
    '3100_3454f_2024-04-08_16-41-19': 'A2M2E',
    '3100_46a3a_2024-04-10_08-07-47': 'aA2B2E',
    '3100_8350a_2024-04-10_07-01-10': 'aA2E',
    '7192_92d91_2024-04-08_14-55-27': 'A2E',
    '7192_11a77_2024-04-08_15-33-22': 'A2B2E',
    '7192_30f3a_2024-04-08_16-14-29': 'A2M2E',
    '7192_a8c1f_2024-04-09_16-54-50': 'aA2B2E',
    '7192_69ba9_2024-04-09_16-54-31': 'aA2E'
}

models = ['A2E', 'aA2E', 'A2B2E', 'aA2B2E', 'A2M2E']
colors = ['cornflowerblue', 'chocolate', 'mediumseagreen']
label_list = ['valence', 'energy', 'tension', 'anger', 'fear', 'happy', 'sad', 'tender']


def opts_parser():
    """ Command line argument setup. """
    parser = argparse.ArgumentParser(description="Script to process adversarial log files.")
    parser.add_argument('--label_vs_pred', action='store_true', default=False, help="Plot label-vs-prediction plot.")
    parser.add_argument('--difference', action='store_true', default=False,
                        help='Plot difference in performance before and after attack.')
    return parser


def set_font_sizes_preds():
    """ Sets the font sizes for the plot showing predicted vs. true values. """
    plt.rc('font', size=17)  # controls default text sizes
    plt.rc('axes', titlesize=17)  # fontsize of the axes title
    plt.rc('axes', labelsize=17)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=17)  # legend fontsize


def get_true_and_pred(log_file, adversarial=False):
    """ Reads given log file, returns true label and adversarial prediction if adversarial=True, else original prediction. """
    log = pd.read_csv(log_file)

    true_labels = convert_strings_to_array(log['true-label'])
    if adversarial:
        preds = convert_strings_to_array(log['new-pred'])
    else:
        preds = convert_strings_to_array(log['orig-pred'])

    return true_labels, preds


def plot_label_vs_prediction(title, label_ids, true_orig_a2b2e, pred_orig_a2b2e, true_adv_a2b2e, pred_adv_a2b2e,
                             true_orig_a2m2e, pred_orig_a2m2e, true_adv_a2m2e, pred_adv_a2m2e,
                             true_orig_aa2b2e, pred_orig_aa2b2e, true_adv_aa2b2e, pred_adv_aa2b2e):
    """
    Plots (and stores) original vs. adversarial predictions and their labels.

    Parameters
    ----------
    title : str
        Title of the plot.
    label_ids : list of int
        List of integer values denoting the labels indices which should be plotted.
    true_orig_a2b2e : ndarray
        True emotion annotations of original files for the blackbox model (nr samples x 8).
    pred_orig_a2b2e : ndarray
        Emotion predictions on original files for the blackbox model (nr samples x 8).
    true_adv_a2b2e : ndarray
        True emotion annotations of adversarial files for the blackbox model (nr samples x 8).
    pred_adv_a2b2e : ndarray
        Emotion predictions on adversarial files for the blackbox model (nr samples x 8).
    true_orig_a2m2e : ndarray
        True emotion annotations of original files for the interpretable model (nr samples x 8).
    pred_orig_a2m2e : ndarray
        Emotion predictions on original files for the interpretable model (nr samples x 8).
    true_adv_a2m2e : ndarray
        True emotion annotations of adversarial files for the interpretable model (nr samples x 8).
    pred_adv_a2m2e : ndarray
        Emotion predictions on adversarial files for the interpretable model (nr samples x 8).
    true_orig_aa2b2e : ndarray
        True emotion annotations of original files for the adversarially trained blackbox model (nr samples x 8).
    pred_orig_aa2b2e : ndarray
        Emotion predictions on original files for the adversarially trained blackbox model (nr samples x 8).
    true_adv_aa2b2e : ndarray
        True emotion annotations of adversarial files for the adversarially trained blackbox model (nr samples x 8).
    pred_adv_aa2b2e : ndarray
        Emotion predictions on adversarial files for the adversarially trained blackbox model (nr samples x 8).
    """
    set_font_sizes_preds()
    fig, axs = plt.subplots(len(label_ids), 3, sharex=True, sharey=True, )
    fig.set_size_inches(14, 4 * len(label_ids))

    for i, idx in enumerate(label_ids):
        assert min(true_orig_a2b2e[:, idx]) == min(true_orig_a2m2e[:, idx]) == min(true_orig_aa2b2e[:, idx])
        assert max(true_orig_a2b2e[:, idx]) == max(true_orig_a2m2e[:, idx]) == max(true_orig_aa2b2e[:, idx])
        min_gt, max_gt = 0, 1

        axs[i, 0].scatter(true_orig_a2b2e[:, idx], pred_orig_a2b2e[:, idx], s=150, c=colors[0], marker='x', lw=2)
        axs[i, 0].scatter(true_adv_a2b2e[:, idx], pred_adv_a2b2e[:, idx], s=150, edgecolors=colors[1],
                          marker='o', facecolors='none', lw=2)
        axs[i, 0].plot([min_gt, max_gt], [min_gt, max_gt], c=colors[2], ls='--')
        left, bottom, width, height = [0.67, 0.03, 0.3, 0.3]
        inset_ax = axs[i, 0].inset_axes([left, bottom, width, height], xticklabels=[], yticklabels=[])
        inset_ax.scatter(true_orig_a2b2e[:, idx], pred_orig_a2b2e[:, idx], s=50, c=colors[0], marker='x', lw=1.5)
        inset_ax.plot([min_gt, max_gt], [min_gt, max_gt], c=colors[2], ls='--')

        axs[i, 1].scatter(true_orig_a2m2e[:, idx], pred_orig_a2m2e[:, idx], s=150, c=colors[0], marker='x', lw=2)
        axs[i, 1].scatter(true_adv_a2m2e[:, idx], pred_adv_a2m2e[:, idx], s=150, edgecolors=colors[1],
                          marker='o', facecolors='none', lw=2)
        axs[i, 1].plot([0, 1], [0, 1], c=colors[2], ls='--')
        inset_ax = axs[i, 1].inset_axes([left, bottom, width, height], xticklabels=[], yticklabels=[])
        inset_ax.scatter(true_orig_a2m2e[:, idx], pred_orig_a2m2e[:, idx], s=50, c=colors[0], marker='x', lw=1.5)
        inset_ax.plot([min_gt, max_gt], [min_gt, max_gt], c=colors[2], ls='--')

        axs[i, 2].scatter(true_orig_aa2b2e[:, idx], pred_orig_aa2b2e[:, idx], s=150, c=colors[0], marker='x',
                          lw=2, label='original')
        axs[i, 2].scatter(true_adv_aa2b2e[:, idx], pred_adv_aa2b2e[:, idx], s=150, edgecolors=colors[1],
                          marker='o', facecolors='none', lw=2, label='adversarial')
        axs[i, 2].plot([0, 1], [0, 1], c=colors[2], ls='--')
        inset_ax = axs[i, 2].inset_axes([left, bottom, width, height], xticklabels=[], yticklabels=[])
        inset_ax.scatter(true_orig_aa2b2e[:, idx], pred_orig_aa2b2e[:, idx], s=50, c=colors[0], marker='x', lw=1.5)
        inset_ax.plot([min_gt, max_gt], [min_gt, max_gt], c=colors[2], ls='--')
        if i == 0:
            axs[i, 0].set_title('A2B2E\n\n' + label_list[idx])
            axs[i, 1].set_title('A2M2E\n\n' + label_list[idx])
            axs[i, 2].set_title('aA2B2E\n\n' + label_list[idx])
        else:
            axs[i, 0].set_title(label_list[idx])
            axs[i, 1].set_title(label_list[idx])
            axs[i, 2].set_title(label_list[idx])

    axs[0, 2].legend(prop={'size': 14})

    fig.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.07, left=0.03, right=0.97, top=0.97)
    plt.suptitle(title)
    fig.supylabel('Prediction')
    fig.supxlabel('True emotion annotations')
    plt.tight_layout()
    plt.savefig(Path(HOME_DIR) / 'plots/true_vs_pred.png', dpi=450)
    plt.close()


def prep_label_vs_prediction():
    """ Reads necessary log files for plot of original vs. adversarial predictions, then calls function to plot scatter-plots. """
    adv_path = Path(RUN_DIR) / 'adversaries'
    # choose emotions that are plotted, here: 2: 'tension', 7: 'tender'
    # for other emotions, use: 0: 'valence', 1: 'energy', 3: 'anger', 4: 'fear', 5: 'happy', 6: 'sad'
    label_ids = [2, 7]

    true_orig_a2b2e, pred_orig_a2b2e = get_true_and_pred(
        adv_path / '21_0a029_2024-03-23_08-45-22_0.001_corr_0.002_test' / 'log.csv')
    true_adv_a2b2e, pred_adv_a2b2e = get_true_and_pred(
        adv_path / '21_0a029_2024-03-23_08-45-22_0.001_corr_0.002_test' / 'log.csv', True)

    true_orig_a2m2e, pred_orig_a2m2e = get_true_and_pred(
        adv_path / '21_71ec2_2024-03-21_17-28-00_0.001_corr_0.002_test' / 'log.csv')
    true_adv_a2m2e, pred_adv_a2m2e = get_true_and_pred(
        adv_path / '21_71ec2_2024-03-21_17-28-00_0.001_corr_0.002_test' / 'log.csv', True)

    true_orig_aa2b2e, pred_orig_aa2b2e = get_true_and_pred(
        adv_path / '21_065d8_2024-04-05_16-07-33_0.001_corr_0.002_test' / 'log.csv')
    true_adv_aa2b2e, pred_adv_aa2b2e = get_true_and_pred(
        adv_path / '21_065d8_2024-04-05_16-07-33_0.001_corr_0.002_test' / 'log.csv', True)

    plot_label_vs_prediction('', label_ids, true_orig_a2b2e, pred_orig_a2b2e, true_adv_a2b2e, pred_adv_a2b2e,
                             true_orig_a2m2e, pred_orig_a2m2e, true_adv_a2m2e, pred_adv_a2m2e,
                             true_orig_aa2b2e, pred_orig_aa2b2e, true_adv_aa2b2e, pred_adv_aa2b2e)


def set_font_sizes_diff():
    """ Sets the font sizes for the plot showing differences in performance after - before attack. """
    plt.rc('font', size=15)  # controls default text sizes
    plt.rc('axes', titlesize=15)  # fontsize of the axes title
    plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=15)  # legend fontsize


def plot_boxplot_difference(metric):
    """
    Plots (and stores) performance differences of different models in `metric`.

    Parameters
    ----------
    metric : dict
        Dictionary containing `model: mae_difference` for different model types.
    """
    set_font_sizes_diff()
    ylim = [3.9, 4.5]
    ylim2 = [0, 3.1]
    ylimratio = (ylim[1] - ylim[0]) / (ylim2[1] - ylim2[0] + ylim[1] - ylim[0])
    ylim2ratio = (ylim2[1] - ylim2[0]) / (ylim2[1] - ylim2[0] + ylim[1] - ylim[0])
    gs = gridspec.GridSpec(2, 1, height_ratios=[ylimratio, ylim2ratio])
    fig = plt.figure()
    fig.set_size_inches(9, 6)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    fig.subplots_adjust(hspace=0.05, top=0.95, bottom=0.05)

    # plot after values
    for idx, arch in enumerate(models):
        diff = [metric[model] for model in metric.keys() if experiments_map[model] == arch]

        print(len(diff))

        ax1.boxplot([diff], vert=True, patch_artist=True, positions=[idx])
        boxplot = ax2.boxplot([diff], vert=True, patch_artist=True, positions=[idx], labels=[arch], widths=[0.65])
        for patch in boxplot['boxes']:
            if arch == 'A2E' or arch == 'A2B2E':
                color = colors[1]
            elif arch == 'aA2E' or arch == 'aA2B2E':
                color = colors[0]
            else:
                color = colors[2]
            patch.set_facecolor(color)
        for median in boxplot['medians']:
            median.set_color('thistle')
            median.set_linewidth(3)

    # create split for A2B2E outlier
    ax1.set_ylim(ylim)  # outlier
    ax2.set_ylim(ylim2)  # rest of data
    ax1.set_title('Difference in Performance')
    ax1.yaxis.set_ticks([4, 4.5])
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    plt.ylabel('$\Delta$MAE')
    ax1.xaxis.set_ticks([], [])
    ax1.xaxis.set_ticks_position('none')
    plt.tight_layout()
    plt.savefig(Path(HOME_DIR) / 'plots/diff_performance.png', dpi=300)
    plt.close()


def prep_difference():
    """ Reads all necessary log files for plot of performance differences, then calls function to plot box-plots. """
    adv_path = Path(RUN_DIR) / 'adversaries'

    metric = {m: None for m in experiments_map.keys()}

    for model_key in experiments_map.keys():
        dir = list(glob(str(adv_path / '{}*/').format(model_key)))
        assert len(dir) == 1
        log_file = Path(dir[0]) / 'log.csv'
        orig_metrics, new_metrics, db_stats, norm_stats, epoch_stats = process_log_file(log_file)

        metric.update({model_key: new_metrics['mae'] - orig_metrics['mae']})

    plot_boxplot_difference(metric)


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()

    if opts.label_vs_pred:
        prep_label_vs_prediction()
    if opts.difference:
        prep_difference()


if __name__ == '__main__':
    main()
