import numpy as np
import matplotlib.pyplot as plt
from math import floor

from sklearn.metrics import classification_report


def display_hyperparameters_search_results(probleme_type, clf, scoring, X_test, y_test, parameters, filter_outliers=False, outliers_thresh=3.5):
    if probleme_type == 'regression':
        results = clf.cv_results_

        for scorer in zip(sorted(scoring)):
            print()
            print(f"*********** Hyper parameters search for {scorer} ***********")
            print(f'*** Best parameter(for scorer in refit parameter): ', clf.best_params_)
            print(f'*** Best score (for scorer in refit parameter): ', clf.best_score_)

            y_true, y_pred = y_test, clf.predict(X_test)
        print()
        print('Best estimator: ', clf.best_estimator_)

        print(results)
        l = len(parameters)
        ncols = 2
        nrows = floor(l / 2) + 1
        fig = plt.figure()
        fig.suptitle("RandomizedSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
        i = 1
        # at each iteration, i.e. for each parameter that has been cross-validated, we create a new subplot
        for parameter in parameters:
            # position of the subplot
            position = nrows * 100 + ncols * 10 + i
            ax = fig.add_subplot(position)
            parameter_values = sorted(results[f"param_{parameter}"].data)
            ax.set_xlabel(parameter)
            ax.set_ylabel("CV score +/- Std")
            # ax.set_ylim(-15, 0)
            # for each score in scorer, we plot the score obtained by the current parameter grid
            for scorer, color in zip(sorted(scoring), ['b', 'k']):
                # we plot one curve for the scores obtained on each set
                for sample, style in (('train', '--'), ('test', '-')):
                    try:
                        sample_score_mean = results[f'mean_{sample}_{scorer}']
                        sample_score_std = results[f'std_{sample}_{scorer}']
                    except:
                        sample_score_mean = results[f'mean_{sample}_score']
                        sample_score_std = results[f'std_{sample}_score']

                    if filter_outliers:
                        outliers_index = ~is_outlier(sample_score_mean, thresh=outliers_thresh)
                        sample_score_mean = sample_score_mean[outliers_index]
                        sample_score_std = sample_score_std[outliers_index]
                        parameter_values_filtered = np.array(parameter_values)[outliers_index]
                    else:
                        parameter_values_filtered = parameter_values

                    ax.fill_between(parameter_values_filtered, sample_score_mean - 2 * sample_score_std,
                                    sample_score_mean + 2 * sample_score_std,
                                    alpha=0.1 if sample == 'test' else 0, color=color)

                    ax.plot(parameter_values_filtered, sample_score_mean, style, color=color,
                            alpha=1 if sample == 'test' else 0.7,
                            label='%s (%s)' % (scorer, sample))

                try:
                    best_index = np.nonzero(results[f'rank_test_{scorer}'] == 1)[0][0]
                    best_score = results[f'mean_test_{scorer}'][best_index]
                except:
                    best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
                    best_score = results['mean_test_score'][best_index]
                best_param = parameter_values[best_index]

                # Plot a dotted vertical line at the best score for that scorer marked by x
                ylim = ax.get_ylim()
                ax.plot([parameter_values[best_index], ] * 2, [ylim[0], best_score],
                        linestyle='-.', color='red', markeredgewidth=3, ms=8)
                # ax.axvline(x=parameter_values[best_index], ymin=best_score, linestyle='-.', color='r', marker='x', markeredgewidth=3, ms=8)

                # Annotate the best score and the best parameter for that scorer
                ax.annotate(f"{np.round(best_score, 2)}",
                            (best_param, best_score + 0.005))

                ax.annotate(f"{best_param}",
                            (best_param, ylim[0] + 0.005))

            plt.legend(loc="best")
            plt.grid(False)

            i += 1
        plt.show()

    elif probleme_type == 'classification':

                # PIck best hyper parameter
        results = clf.cv_results_
        for scorer in zip(sorted(scoring)):
            print()
            print(scorer)
            print('Best parameter(for scorer in refit parameter): ', clf.best_params_)
            print('Best score (for scorer in refit parameter): ', clf.best_score_)
            print()
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            print("Grid scores on development set:")

            means = results['mean_test_%s' % scorer]
            stds = results['std_test_%s' % scorer]
            for mean, std, params in zip(means, stds, results['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
        print()

        print()

        print('Best estimator: ', clf.best_estimator_)

        results = clf.cv_results_
        print(results)
        # Récupère la valeur des paramètres et change le type du nparray qui pose un problème dans matplotlib
        params = results['param_bagging__base_estimator__C'].data.astype('str').astype('float')

        fig, ax = plt.subplots(figsize=(13, 13))
        ax.set_title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)
        ax.set_xlabel("min_samples_split")
        ax.set_ylabel("CV score +/- Std")

        for scorer, color in zip(sorted(scoring), ['b', 'k']):
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                sample_score_std = results['std_%s_%s' % (sample, scorer)]

                ax.fill_between(params, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0, color=color)

                ax.plot(params, sample_score_mean, style, color=color,
                        alpha=1 if sample == 'test' else 0.7,
                        label='%s (%s)' % (scorer, sample))

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]
            best_param = params[best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([params[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score and the best parameter for that scorer
            ax.annotate("%0.2f" % best_score,
                        (best_param, best_score + 0.005))

            ax.annotate("%0.2f" % best_param,
                        (best_param, 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        plt.show()


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def two_scales(ax1, time, data1, data2, c1, c2, feature):

    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('fft')

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel(feature)
    return ax1, ax2


def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


def plot_ttf_vs_feature(X, y, feature):
    # plot y_pred vs y for the train and test sets
    X_plot = X.copy()
    X_plot['y'] = y
    X_plot.sort_index(inplace=True)

    # Create axes
    fig, ax = plt.subplots(figsize=(18, 8))
    ax1, ax2 = two_scales(ax, X_plot.index, X_plot['y'], X_plot[feature], 'r', 'b', feature)

    # Change color of each axis
    color_y_axis(ax1, 'r')
    color_y_axis(ax2, 'b')
    ax.legend(loc=(1, 0.5))

    correlation = np.corrcoef(X_plot[feature], X_plot['y'])[0][1]
    ax.set_title(f'correlation : {correlation}')
    plt.tight_layout()
    plt.show()

# ====================================================
# Plot the effect of denoising and filtering
# ====================================================


import data_preparation as dp
import pandas as pd


def plot_denoising_filtering_effects():
    SIGNAL_LEN = 150000
    seismic_signals = pd.read_csv("input/train.csv", nrows=450000, dtype={'acoustic_data': np.int16,
                                                                          'time_to_failure': np.float32})
    data_len = len(seismic_signals)
    acoustic_data = seismic_signals.acoustic_data
    time_to_failure = seismic_signals.time_to_failure

    signals = []
    targets = []
    for i in range(data_len // SIGNAL_LEN):
        print(i)
        min_lim = SIGNAL_LEN * i
        max_lim = min([SIGNAL_LEN * (i + 1), data_len - 1])
        print(max_lim)

        signals.append(list(acoustic_data[min_lim: max_lim]))
        targets.append(time_to_failure[max_lim])

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(30, 15))

    ax[0, 0].plot(signals[0], 'crimson')
    ax[0, 0].set_title('Original Signal', fontsize=16)
    ax[0, 1].plot(dp.high_pass_filter(signals[0], low_cutoff=10000, SAMPLE_RATE=4000000), 'mediumvioletred')
    ax[0, 1].set_title('After High-Pass Filter', fontsize=16)
    ax[0, 2].plot(dp.denoise_signal(signals[0]), 'darkmagenta')
    ax[0, 2].set_title('After Wavelet Denoising', fontsize=16)
    ax[0, 3].plot(dp.denoise_signal(dp.high_pass_filter(signals[0], low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1), 'indigo')
    ax[0, 3].set_title('After High-Pass Filter and Wavelet Denoising', fontsize=16)

    ax[1, 0].plot(signals[1], 'crimson')
    ax[1, 0].set_title('Original Signal', fontsize=16)
    ax[1, 1].plot(dp.high_pass_filter(signals[1], low_cutoff=10000, SAMPLE_RATE=4000000), 'mediumvioletred')
    ax[1, 1].set_title('After High-Pass Filter', fontsize=16)
    ax[1, 2].plot(dp.denoise_signal(signals[1]), 'darkmagenta')
    ax[1, 2].set_title('After Wavelet Denoising', fontsize=16)
    ax[1, 3].plot(dp.denoise_signal(dp.high_pass_filter(signals[1], low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1), 'indigo')
    ax[1, 3].set_title('After High-Pass Filter and Wavelet Denoising', fontsize=16)

    ax[2, 0].plot(signals[2], 'crimson')
    ax[2, 0].set_title('Original Signal', fontsize=16)
    ax[2, 1].plot(dp.high_pass_filter(signals[2], low_cutoff=10000, SAMPLE_RATE=4000000), 'mediumvioletred')
    ax[2, 1].set_title('After High-Pass Filter', fontsize=16)
    ax[2, 2].plot(dp.denoise_signal(signals[2]), 'darkmagenta')
    ax[2, 2].set_title('After Wavelet Denoising', fontsize=16)
    ax[2, 3].plot(dp.denoise_signal(dp.high_pass_filter(signals[2], low_cutoff=10000, SAMPLE_RATE=4000000), wavelet='haar', level=1), 'indigo')
    ax[2, 3].set_title('After High-Pass Filter and Wavelet Denoising', fontsize=16)

    plt.show()
