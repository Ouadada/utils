import pandas as pd
import numpy as np
from tqdm import tqdm  # for displaying a progression bar in the loops
import pywt
from collections import Counter

from scipy.stats import kurtosis, skew, trim_mean, entropy, hmean, gmean, kstat, moment, kstatvar, pearsonr
from scipy.signal import hilbert, hann, convolve, welch, sosfilt, lfilter
from scipy.fftpack import fft

from sklearn.linear_model import LinearRegression

from tsfresh.feature_extraction import feature_calculators

from joblib import Parallel, delayed


'''
train = pd.read_csv("input/train.csv", nrows=None, dtype={'acoustic_data': np.int16,
                                                          'time_to_failure': np.float32})

overlapping_indexes = np.array([37, 333, 697, 925, 1250, 1457, 1638, 2052, 2255, 2502, 2795, 3078, 3305, 3525, 3903, 4146])

for i in overlapping_indexes:
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(train['acoustic_data'].values[i * 150000:(i + 1) * 150000], color='g', label='acoustic_data')
    ax2 = ax1.twinx()

    ax2.plot(train['time_to_failure'].values[i * 150000:(i + 1) * 150000], color='r', label='ttf')
    fig.tight_layout()
    plt.show()

from tqdm import tqdm  # for displaying a progression bar in the loops
segments = int(np.floor(train.shape[0] / (ROWS / 5))) - 5
seg_list = []
for segment in tqdm(range(segments)):
    # select the rows of train corresponding to the segment
    index_min = int(segment * ROWS / 5)
    index_max = int(segment * ROWS / 5 + ROWS)
    seg = train.iloc[index_min: index_max]
    x_seg = pd.Series(seg['acoustic_data'].values)
    # the target value of the segment is the time_to_failure of the last row
    ttf_0 = seg['time_to_failure'].values[0]
    ttf_1 = seg['time_to_failure'].values[-1]
    if ttf_0 < ttf_1:
        # seg_list.append((segment, ttf_0 - ttf_1, ttf_0, ttf_1))
        seg_list.append(segment)
print(seg_list)
del train

import sys
sys.exit(0)
'''

# ====================================
# Drop highly correlated features
# ===================================
def drop_highly_correlated_features(X, X_sub, threshold=0.95):

    print("Deleting features highly correlated with an other one...")
    corr_matrix = X.corr().abs()
    # Select upper triangle of correlation matrix
    upper_coor_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of features columns with correlation greater than threshold
    to_drop = [column for column in upper_coor_matrix.columns if any(upper_coor_matrix[column] > threshold)]
    X = X.drop(to_drop, axis=1)
    X_sub = X_sub.drop(to_drop, axis=1)

    return X, X_sub


def drop_poorly_correlated_features(X, X_sub, y, use_p_value, threshold=0.4):

    print("Deleting features poorly correlated with the target...")
    
    if not use_p_value:
        X['target'] = y.values
        corr_matrix = X.corr().abs()
        # Select upper triangle of correlation matrix
        lower_coor_matrix = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        y_coor_matrix = lower_coor_matrix.sort_values(['target'], ascending=False).loc[['target'], :]

        # Find index of features columns with correlation lesser than threshold
        to_drop = [column for column in y_coor_matrix.columns if any(y_coor_matrix[column] < threshold)]
        X.drop(['target'], axis=1, inplace=True)
        X.drop(to_drop, axis=1, inplace=True)
        X_sub.drop(to_drop, axis=1, inplace=True)

    else:

        pcol = []
        pcor = []
        pval = []

        for col in X.columns:
            pcol.append(col)
            pcor.append(abs(pearsonr(X[col].values, y.values.ravel())[0]))
            pval.append(abs(pearsonr(X[col].values, y.values.ravel())[1]))

        df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
        df.sort_values(by=['cor', 'pval'], inplace=True)
        df.dropna(inplace=True)
        df = df.loc[df['pval'] <= 0.05]

        drop_cols = []

        for col in X.columns:
            if col not in df['col'].tolist():
                drop_cols.append(col)

        X.drop(labels=drop_cols, axis=1, inplace=True)
        X_sub.drop(labels=drop_cols, axis=1, inplace=True)

    return X, X_sub



def filter_outliers(X, y, low=0.01):

  high_quantile = 1 - low_quantile
  quant_X = X.quantile([low_quantile, high_quantile], axis=0)
  print(quant_X.transpose())

  X_filtered = X.copy()
  X_filtered = X_filtered.apply(lambda x: x[(x > quant_X.loc[low, x.name]) &
                                            (x < quant_X.loc[high, x.name])], axis=0)
  print(X_filtered.describe())
  X_filtered.dropna(inplace=True)

  X = X_filtered.copy()
  y = y.ix[X.index]

  return X, y




def create_earthquake_category(X, y):
    '''
    Create a category in X that tells which of the 16 earthquake the observation is part of
    '''
    quake_cat = 1
    quake_cat_array = []
    prev_fft = y[0]

    for i in range(X.shape[0]):
        fft = y[i]
        if fft <= prev_fft:
            quake_cat_array.append(quake_cat)
        else:
            quake_cat += 1
            quake_cat_array.append(quake_cat)
        prev_fft = fft

    return quake_cat_array


    '''
    # an other quite elegant method
    diff = y.diff()
    print('1', diff)
    # get the index of the observations just before a quake happens
    quake_position = np.where(diff > 0)[0]
    print('2', quake_position)
    # calculates the length between two quakes
    quake_interval = np.diff(quake_position)
    print('3', quake_interval)
    # add the index of the first quake to complete the intervals
    quake_interval = np.append(quake_position[0], quake_interval)
    print('4', quake_interval)
    # create one array for each quake. Each array is of the size of the number of observations between the previous quake and the quake and contains the number of the quake
    quake_idx = [np.repeat(i, quake_interval[i]) for i in range(len(quake_interval))]
    print('5', quake_idx)
    # put all arrays together
    quake_idx = np.concatenate(quake_idx)
    print('6', quake_idx)
    # add missing observations in the end
    quake_idx = np.append(quake_idx, np.repeat(15, len(y) - len(quake_idx)))
    print('7', quake_idx)
    '''


def resample(type, X, y, quake_cat_array, resample_size=24000):
    '''
    Create bins of continous target and apply a SMOTE oversampling so each bin has the same frequency
    '''
    if type == 'SMOTE':
        X['time_to_failure'] = y
        X['quake_cat_array'] = quake_cat_array
        X['quake_cat_array'] = X['quake_cat_array'].astype('category')


        X_cols = X.columns  # Save columns names of X_train because SMOTE return ndarrays
        y = pd.DataFrame(y, columns=['time_to_failure'])

        # Creates categories from target values
        y['time_to_failure'] = pd.cut(y.time_to_failure, [0, 2, 10, 12, 14, 20], labels=['a', 'b', 'c', 'd', 'e'])
        # Print number of each categories
        print(y.groupby('time_to_failure')['time_to_failure'].agg(['count']))

        # Resampling with SMOTE method
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=12, )
        X, y = sm.fit_sample(X, y)

        X = pd.DataFrame(X, columns=X_cols)
        y_train_df = pd.DataFrame(y, columns=['time_to_failure'])
        #print(y_train_df.groupby('time_to_failure')['time_to_failure'].agg(['count']))

        y = X['time_to_failure']
        quake_cat_array = np.floor(X['quake_cat_array'].values)
        X.drop(['time_to_failure', 'quake_cat_array'], axis=1, inplace=True)

    elif type == 'random':
        # Disable warning for copying slices 
        pd.options.mode.chained_assignment = None  # default='warn'

        #X = np.append(X, y, axis=1)
        #X = np.append(X, quake_cat_array, axis=1)
        X_temp = X.copy()
        X_temp['time_to_failure'] = y
        X_temp['quake_cat_array'] = quake_cat_array

        # X_count gives for each slice the indexes in which draw or reample indexes.
        X_counts = X_temp.groupby('quake_cat_array')['quake_cat_array'].agg(['count'])

        # X_count_resampled gives us how many indexes of each slice we must draw
        X_counts_resampled = X_counts.apply(lambda x: np.round(resample_size * x / x.sum(), 0))
        X_counts_resampled['count_int'] = X_counts_resampled['count'].apply(int)

        min_index = 0
        max_index = 0
        X_counts_lis_temp = X_counts.cumsum().values.ravel()
        X_counts_list = np.concatenate(([0], X_counts_lis_temp))
        indexes = []

        for i in range(len(X_counts_list) - 1):
            min_index = X_counts_list[i]
            max_index = X_counts_list[i + 1]
            indexes += list(np.random.randint(min_index, max_index, size=X_counts_resampled['count_int'].iloc[i]))
       
        X_result = X_temp.iloc[indexes]
        y_result = X_result['time_to_failure'].values
        quake_cat_array_result = X_result['quake_cat_array'].values
        X_result.drop(['time_to_failure', 'quake_cat_array'], axis=1, inplace=True)

        del X_temp

    # enaable warning for copying slices 
    pd.options.mode.chained_assignment = 'warn'
    return X_result, y_result, quake_cat_array_result


def clean_nan_inf(X, y, action='delete'):
    '''
    Check if any nan are in the dataset and delete the column or replace the by the mean value depending on the value of 'action'
    '''
    print(f"Taking care ({action}) of nan and inf...")
    if action == 'delete':

        idx = ~X.isin([np.nan, np.inf, -np.inf]).any(1)
        X = X[idx]
        y = y[idx]

        return X, y

    elif action == 'replace':
        means_dict = {}
        for col in X.columns:
            if X[col].isnull().any():
                mean_value = X.loc[X[col] != -np.inf, col].mean()
                X.loc[X[col] == -np.inf, col] = mean_value
                X[col] = X[col].fillna(mean_value)
                means_dict[col] = mean_value
        return X, y


def check_missing_data(df):
    flag = df.isna().sum().any()
    if flag:
        total = df.isnull().sum()
        percent = (df.isnull().sum()) / (df.isnull().count() * 100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        output = output[output.Total > 0]
        return output
    else:
        return "No missing data"


def create_y(y, y_seg, segment):

    y.loc[segment, 'time_to_failure'] = y_seg

    return y


# =====================================================================================
# Creation of train and test datasets and submission file
# =====================================================================================


class FeatureGenerator(object):
    def __init__(self, dtype, nrows=None, n_jobs=1, chunk_size=None):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.nrows = nrows
        self.n_jobs = n_jobs
        self.filename = None
        self.test_files = []
        if self.dtype == 'train':
            self.filename = 'input/train.csv'
            self.total_data = int(629145491) / self.chunk_size
        else:
            submission = pd.read_csv('input/sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, 'input/test/' + seg_id + '.csv'))
                self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, nrows=self.nrows, iterator=True, chunksize=self.chunk_size,
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                del df
                yield seg_id, x, y  # yield is like return but instead returns a generator
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values[-self.chunk_size:]
                del df
                yield seg_id, x, -999


    def generate(self):
        feature_list = []
        # for s, x, y in tqdm(self.read_chunks(), total=self.total_data):
        #     self.get_features(x, y, s)
        res = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.get_features)(x, y, s)
                                                                for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)

    def get_features(self, x, y, seg_id):
        features_dict = dict()
        features_dict['target'] = y
        features_dict['seg_id'] = seg_id

        x = pd.Series(x)
        x_seg = x.copy()
        del x
        x_seg_centered = x_seg - np.mean(x_seg)

        b, a = get_filter_coef(btype='lowpass', cutoff=18000)
        x_seg_hp_filtered = lfilter(b, a, x_seg_centered)

        zc = np.fft.fft(x_seg_hp_filtered)
        zc = zc[:20000]

        # FFT transform values
        realFFT = np.real(zc)
        imagFFT = np.imag(zc)

        MAX_FREQ_IDX = 20000
        FREQ_STEP = 2500

        freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]
        magFFT = np.abs(zc)
        phzFFT = np.angle(zc)
        phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
        phzFFT[phzFFT == np.inf] = np.pi / 2.0
        phzFFT = np.nan_to_num(phzFFT)

        for freq in freq_bands:
            features_dict[f'FFT_Mag_01q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
            features_dict[f'FFT_Mag_10q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
            features_dict[f'FFT_Mag_90q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
            features_dict[f'FFT_Mag_99q_{freq}'] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
            features_dict[f'FFT_Mag_mean_{freq}'] = np.mean(magFFT[freq: freq + FREQ_STEP])
            features_dict[f'FFT_Mag_std_{freq}'] = np.std(magFFT[freq: freq + FREQ_STEP])
            features_dict[f'FFT_Mag_max_{freq}'] = np.max(magFFT[freq: freq + FREQ_STEP])

            features_dict[f'FFT_Phz_mean_{freq}'] = np.mean(phzFFT[freq: freq + FREQ_STEP])
            features_dict[f'FFT_Phz_std_{freq}'] = np.std(phzFFT[freq: freq + FREQ_STEP])

        features_dict['FFT_Rmean'] = realFFT.mean()
        features_dict['FFT_Rstd'] = realFFT.std()
        features_dict['FFT_Rmax'] = realFFT.max()
        features_dict['FFT_Rmin'] = realFFT.min()
        features_dict['FFT_Imean'] = imagFFT.mean()
        features_dict['FFT_Istd'] = imagFFT.std()
        features_dict['FFT_Imax'] = imagFFT.max()
        features_dict['FFT_Imin'] = imagFFT.min()

        features_dict['FFT_Rmean_first_6000'] = realFFT[:6000].mean()
        features_dict['FFT_Rstd__first_6000'] = realFFT[:6000].std()
        features_dict['FFT_Rmax_first_6000'] = realFFT[:6000].max()
        features_dict['FFT_Rmin_first_6000'] = realFFT[:6000].min()
        features_dict['FFT_Rmean_first_18000'] = realFFT[:18000].mean()
        features_dict['FFT_Rstd_first_18000'] = realFFT[:18000].std()
        features_dict['FFT_Rmax_first_18000'] = realFFT[:18000].max()
        features_dict['FFT_Rmin_first_18000'] = realFFT[:18000].min()

        b, a = get_filter_coef(btype = 'lowpass', cutoff=2500)
        xc0 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=2500, high=5000)
        xc1 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=5000, high=7500)
        xc2 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=7500, high=10000)
        xc3 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=10000, high=12500)
        xc4 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=12500, high=15000)
        xc5 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=15000, high=17500)
        xc6 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'bandpass', low=17500, high=20000)
        xc7 = lfilter(b, a, x_seg_centered)

        b, a = get_filter_coef(btype = 'highpass', cutoff=20000)
        xc8 = lfilter(b, a, x_seg_centered)

        sigs = [x_seg, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),
            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]


        # lists with parameters to iterate over them
        percentiles = [0.1, 1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 99.9]
        hann_windows = [50, 150, 1500, 15000]
        #spans = [300, 3000, 30000, 50000]
        windows = [100, 500, 1000, 10000]
        borders = list(range(-4000, 4001, 1000))
        peaks = [10, 20, 50, 100]
        #coefs = [1, 5, 10, 50, 100]
        #lags = [10, 100, 1000, 10000]
        autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]

        for k, sig in enumerate(sigs):

            crossings = calculate_crossings(sig)
            if x_seg.shape[0] > 10000:  # to exclude small wavelet decompositions
                features_dict[f'no_zero_crossings_{k}'] = crossings[0]
                features_dict[f'no_mean_crossings_{k}'] = crossings[1]
                features_dict[f'entropy_{k}'] = calculate_entropy(sig)
                features_dict[f'power_{k}'] = np.sum(sig**2) / 4e6
                features_dict[f'mean_{k}'] = np.nanmean(sig)
                features_dict[f'mean10_{k}'] = trim_mean(sig, 0.1)
                features_dict[f'std_{k}'] = np.nanstd(sig)
                features_dict[f'var_{k}'] = np.nanvar(sig)
                features_dict[f'rms_{k}'] = np.nanmean(np.sqrt(sig**2))
                features_dict[f'mad_{k}'] = np.nanmean(np.absolute(sig - np.nanmean(sig)))
                features_dict[f'kurt_{k}'] = kurtosis(sig)
                features_dict[f'sk_{k}'] = skew(sig)
                features_dict[f'max_{k}'] = np.max(sig)
                features_dict[f'min_{k}'] = np.min(sig)

                # percentiles on original and absolute values
                for p in percentiles:
                    features_dict[f'percentile_{p}_{k}'] = np.nanpercentile(sig, p)
                    features_dict[f'abs_percentile_{p}_{k}'] = np.nanpercentile(np.abs(sig), p)
                features_dict[f'iqr_{k}'] = np.subtract(*np.nanpercentile(sig, [75, 25]))

                # peaks
                for peak in peaks:
                    features_dict[f'num_peaks_{peak}_{k}'] = feature_calculators.number_peaks(sig, peak)

                 # geometric and harminic means
                features_dict[f'hmean_{k}'] = hmean(np.abs(sig[np.nonzero(sig)[0]]))
                features_dict[f'gmean_{k}'] = gmean(np.abs(sig[np.nonzero(sig)[0]]))

                # k-statistic and moments
                for i in range(1, 5):
                    features_dict[f'kstat_{i}_{k}'] = kstat(sig, i)
                    features_dict[f'moment_{i}_{k}'] = moment(sig, i)

                for i in [1, 2]:
                    features_dict[f'kstatvar_{i}_{k}'] = kstatvar(sig, i)

                # autocorrelation
                for autocorr_lag in autocorr_lags:
                    features_dict[f'autocorrelation_{autocorr_lag}_{k}'] = feature_calculators.autocorrelation(sig, autocorr_lag)
                    features_dict[f'c3_{autocorr_lag}_{k}'] = feature_calculators.c3(sig, autocorr_lag)

                # Number of points in the range
                features_dict[f'range_minf_m4000_{k}'] = feature_calculators.range_count(sig, -np.inf, -4000)
                features_dict[f'range_p4000_pinf_{k}'] = feature_calculators.range_count(sig, 4000, np.inf)

                features_dict[f'Hilbert_mean_{k}'] = np.abs(hilbert(sig)).mean()

                for hw in hann_windows:
                  features_dict[f'Hann_window_mean_{hw}_{k}'] = (convolve(sig, hann(150), mode='same') / sum(hann(150))).mean()


                # Number of points in the range
                features_dict[f'range_minf_m4000_{k}'] = feature_calculators.range_count(sig, -np.inf, -4000)
                features_dict[f'range_p4000_pinf_{k}'] = feature_calculators.range_count(sig, 4000, np.inf)

                for i, j in zip(borders, borders[1:]):
                    features_dict[f'range_{i}_{j}_{k}'] = feature_calculators.range_count(sig, i, j)

                features_dict[f'classic_sta_lta1_mean_{k}'] = classic_sta_lta(sig, 500, 10000).mean()
                features_dict[f'classic_sta_lta2_mean_{k}'] = classic_sta_lta(sig, 5000, 100000).mean()
                features_dict[f'classic_sta_lta3_mean_{k}'] = classic_sta_lta(sig, 3333, 6666).mean()
                features_dict[f'classic_sta_lta4_mean_{k}'] = classic_sta_lta(sig, 10000, 25000).mean()
                features_dict[f'classic_sta_lta5_mean_{k}'] = classic_sta_lta(sig, 50, 1000).mean()
                features_dict[f'classic_sta_lta6_mean_{k}'] = classic_sta_lta(sig, 100, 5000).mean()
                features_dict[f'classic_sta_lta7_mean_{k}'] = classic_sta_lta(sig, 333, 666).mean()
                features_dict[f'classic_sta_lta8_mean_{k}'] = classic_sta_lta(sig, 4000, 10000).mean()

                features_dict[f'mean_change_abs_{k}'] = np.mean(np.diff(sig))
                features_dict[f'mean_change_rate_{k}'] = calc_change_rate(sig)
                features_dict[f'abs_max_{k}'] = np.abs(sig).max()
                features_dict[f'abs_min_{k}'] = np.abs(sig).min()
                features_dict[f'std_first_100000_{k}'] = sig[:50000].std()
                features_dict[f'std_last_50000_{k}'] = sig[-50000:].std()
                features_dict[f'std_first_10000_{k}'] = sig[:10000].std()
                features_dict[f'std_last_10000_{k}'] = sig[-10000:].std()
                features_dict[f'ave_first_50000_{k}'] = sig[:50000].mean()
                features_dict[f'ave_last_50000_{k}'] = sig[-50000:].mean()
                features_dict[f'ave_first_10000_{k}'] = sig[:10000].mean()
                features_dict[f'ave_last_10000_{k}'] = sig[-10000:].mean()
                features_dict[f'min_first_50000_{k}'] = sig[:50000].min()
                features_dict[f'min_last_50000_{k}'] = sig[-50000:].min()
                features_dict[f'min_first_10000_{k}'] = sig[:10000].min()
                features_dict[f'min_last_10000_{k}'] = sig[-10000:].min()
                features_dict[f'max_first_50000_{k}'] = sig[:50000].max()
                features_dict[f'max_last_50000_{k}'] = sig[-50000:].max()
                features_dict[f'max_first_10000_{k}'] = sig[:10000].max()
                features_dict[f'max_last_10000_{k}'] = sig[-10000:].max()
                features_dict[f'skew_first_50000_{k}'] = skew(sig[:50000])
                features_dict[f'skew_last_50000_{k}'] = skew(sig[-50000:])
                features_dict[f'skew_first_10000_{k}'] = skew(sig[:10000])
                features_dict[f'skew_last_10000_{k}'] = skew(sig[-10000:])
                features_dict[f'kurt_first_50000_{k}'] = kurtosis(sig[:50000])
                features_dict[f'kurt_last_50000_{k}'] = kurtosis(sig[-50000:])
                features_dict[f'kurt_first_10000_{k}'] = kurtosis(sig[:10000])
                features_dict[f'kurt_last_10000_{k}'] = kurtosis(sig[-10000:])
                features_dict[f'max_to_min_{k}'] = sig.max() / np.abs(sig.min())
                features_dict[f'max_to_min_diff_{k}'] = sig.max() - np.abs(sig.min())
                features_dict[f'count_big_{k}'] = len(sig[np.abs(sig) > 500])
                features_dict[f'sum_{k}'] = sig.sum()
                features_dict[f'mean_change_rate_first_50000_{k}'] = calc_change_rate(sig[:50000])
                features_dict[f'mean_change_rate_last_50000_{k}'] = calc_change_rate(sig[-50000:])
                features_dict[f'mean_change_rate_first_10000_{k}'] = calc_change_rate(sig[:10000])
                features_dict[f'mean_change_rate_last_10000_{k}'] = calc_change_rate(sig[-10000:])

                # trend
                features_dict[f'trend_{k}'] = add_trend_feature(sig)
                features_dict[f'abs_trend_{k}'] = add_trend_feature(sig, abs_values=True)
                features_dict[f'abs_mean_{k}'] = np.abs(sig).mean()
                features_dict[f'abs_std_{k}'] = np.abs(sig).std()

                features_dict[f'Moving_average_400_mean_{k}'] = sig.rolling(window=400).mean().mean(skipna=True)
                features_dict[f'Moving_average_700_mean_{k}'] = sig.rolling(window=700).mean().mean(skipna=True)
                features_dict[f'Moving_average_1500_mean_{k}'] = sig.rolling(window=1500).mean().mean(skipna=True)
                features_dict[f'Moving_average_3000_mean_{k}'] = sig.rolling(window=3000).mean().mean(skipna=True)
                features_dict[f'Moving_average_6000_mean_{k}'] = sig.rolling(window=6000).mean().mean(skipna=True)


                ewma = pd.Series.ewm
                features_dict[f'exp_Moving_average_300_mean_{k}'] = ewma(sig, span=300).mean().mean(skipna=True)
                features_dict[f'exp_Moving_average_3000_mean_{k}'] = ewma(sig, span=3000).mean().mean(skipna=True)
                features_dict[f'exp_Moving_average_6000_mean_{k}'] = ewma(sig, span=6000).mean().mean(skipna=True)
                features_dict[f'exp_Moving_average_30000_mean_{k}'] = ewma(sig, span=30000).mean().mean(skipna=True)

                no_of_std = 2
                features_dict[f'MA_700MA_std_mean_{k}'] = sig.rolling(window=700).std().mean()
                features_dict[f'MA_700MA_BB_high_mean_{k}'] = (features_dict[f'Moving_average_700_mean_{k}'] + no_of_std * features_dict[f'MA_700MA_std_mean_{k}']).mean()
                features_dict[f'MA_700MA_BB_low_mean_{k}'] = (features_dict[f'Moving_average_700_mean_{k}'] - no_of_std * features_dict[f'MA_700MA_std_mean_{k}']).mean()
                features_dict[f'MA_400MA_std_mean_{k}'] = sig.rolling(window=400).std().mean()
                features_dict[f'MA_400MA_BB_high_mean_{k}'] = (features_dict[f'Moving_average_400_mean_{k}'] + no_of_std * features_dict[f'MA_400MA_std_mean_{k}']).mean()
                features_dict[f'MA_400MA_BB_low_mean_{k}'] = (features_dict[f'Moving_average_400_mean_{k}'] - no_of_std * features_dict[f'MA_400MA_std_mean_{k}']).mean()
                features_dict[f'MA_1000MA_std_mean_{k}'] = sig.rolling(window=1000).std().mean()


                
                # computes the same statistics for sub sample of the signal
                # for i in range(3):
                #     sub_x = x[50000 * i : 50000 * (i + 1) -1]
                #     suffix = 50000 * (i + 1) - 1
                #     crossings = calculate_crossings(sub_x)
                #     features_dict[f'no_zero_crossings_{suffix}'] = crossings[0]
                #     features_dict[f'no_mean_crossings_{suffix}'] = crossings[1]
                #     features_dict[f'entropy_{suffix}'] = calculate_entropy(sub_x)
                #     features_dict[f'power_{suffix}'] = np.sum(sub_x**2) / 4e6
                #     features_dict[f'n5_{suffix}'] = np.nanpercentile(sub_x, 5)
                #     features_dict[f'n25_{suffix}'] = np.nanpercentile(sub_x, 25)
                #     features_dict[f'n75_{suffix}'] = np.nanpercentile(sub_x, 75)
                #     features_dict[f'n95_{suffix}'] = np.nanpercentile(sub_x, 95)
                #     features_dict[f'median_{suffix}'] = np.nanpercentile(sub_x, 50)
                #     features_dict[f'mean_{suffix}'] = np.nanmean(sub_x)
                #     features_dict[f'std_{suffix}'] = np.nanstd(sub_x)
                #     features_dict[f'var_{suffix}'] = np.nanvar(sub_x)
                #     features_dict[f'rms_{suffix}'] = np.nanmean(np.sqrt(sub_x**2))
                #     features_dict[f'mad_{suffix}'] = np.nanmean(np.absolute(sub_x - np.nanmean(sub_x)))
                #     features_dict[f'kurt_{suffix}'] = kurtosis(sub_x)
                #     features_dict[f'sk_{suffix}'] = skew(sub_x)
                #     features_dict[f'masub_x'] = np.max(sub_x)
                #     features_dict[f'min_{suffix}'] = np.min(sub_x)
                #     
                #     # percentiles on original and absolute values
                #     for p in percentiles:
                #         features_dict[f'percentile_{p}_{suffix}'] = np.nanpercentile(x, p)
                #         features_dict[f'abs_percentile_{p}_{suffix}'] = np.nanpercentile(np.abs(x), p)
                #         
                #     # peaks
                #     for peak in peaks:
                #         features_dict[f'num_peaks_{peak}_{suffix}'] = feature_calculators.number_peaks(x, peak)
                #         
                #      # geometric and harminic means
                #     features_dict[f'hmean_{suffix}'] = hmean(np.abs(x[np.nonzero(x)[0]]))
                #     features_dict[f'gmean_{suffix}'] = gmean(np.abs(x[np.nonzero(x)[0]]))
                #     
                #     # k-statistic and moments
                #     for i in range(1, 5):
                #         features_dict[f'kstat_{i}_{suffix}'] = kstat(x, i)
                #         features_dict[f'moment_{i}_{suffix}'] = moment(x, i)
                #         
                #     for i in [1, 2]:
                #         features_dict[f'kstatvar_{i}_{suffix}'] = kstatvar(x, i)
                #         
                #     # autocorrelation
                #     for autocorr_lag in autocorr_lags:
                #         features_dict[f'autocorrelation_{autocorr_lag}_{suffix}'] = feature_calculators.autocorrelation(x, autocorr_lag)
                #         features_dict[f'c3_{autocorr_lag}_{suffix}'] = feature_calculators.c3(x, autocorr_lag)
                #         
                #     # Number of points in the range
                #     features_dict[f'range_minf_m4000_{suffix}'] = feature_calculators.range_count(x, -np.inf, -4000)
                #     features_dict[f'range_p4000_pinf_{suffix}'] = feature_calculators.range_count(x, 4000, np.inf)
                
        del x_seg_hp_filtered
        del zc

        for windows in [100, 1000, 10000]:
            if x_seg.shape[0] > windows:
                x_roll_std = x_seg.rolling(windows).std().dropna().values
                x_roll_mean = x_seg.rolling(windows).mean().dropna().values

                features_dict['ave_roll_std_' + str(windows)] = x_roll_std.mean()
                features_dict['std_roll_std_' + str(windows)] = x_roll_std.std()
                features_dict['max_roll_std_' + str(windows)] = x_roll_std.max()
                features_dict['min_roll_std_' + str(windows)] = x_roll_std.min()
                features_dict['q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
                features_dict['q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
                features_dict['q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
                features_dict['q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
                features_dict['av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                features_dict['av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                features_dict['abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
                features_dict['ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                features_dict['std_roll_mean_' + str(windows)] = x_roll_mean.std()
                features_dict['max_roll_mean_' + str(windows)] = x_roll_mean.max()
                features_dict['min_roll_mean_' + str(windows)] = x_roll_mean.min()
                features_dict['q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
                features_dict['q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
                features_dict['q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
                features_dict['q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
                features_dict['av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                with np.errstate(all='ignore'):
                    features_dict['av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                features_dict['abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

        return features_dict


def create_train_set(train, segments, rows, version, use_wavelets, filtering=True, denoising=True, save=True):
    print("Starting creation of train set")
    X = pd.DataFrame(index=range(segments), dtype=np.float64)
    y = pd.DataFrame(index=range(segments), dtype=np.float64,
                     columns=['time_to_failure'])

    if filtering:
        X = high_pass_filter(X, low_cutoff=10000, SAMPLE_RATE=4000000)

    if denoising:
        X = denoise_signal(X, wavelet='db4', level=1)

    i = 1
    for segment in tqdm(range(segments)):
        print(f"Computing segment {i}")
        index_min = int(segment * rows / 1)
        index_max = int(segment * rows / 1 + rows)
        # select the rows of train corresponding to the segment
        seg = train.iloc[index_min: index_max]
        x_seg = pd.Series(seg['acoustic_data'].values)

        # overlapping_indexes = []
        # the target value of the segment is the time_to_failure of the last row
        # ttf_0 = seg['time_to_failure'].values[0]
        # ttf_1 = seg['time_to_failure'].values[-1]
        # if ttf_0 < ttf_1:
        #     overlapping_indexes.append(segment)
        # if
        y_seg = seg['time_to_failure'].values[-1]

        if filtering:
            x_seg = high_pass_filter(x_seg, low_cutoff=10000, SAMPLE_RATE=4000000)
        if denoising:
            x_seg = denoise_signal(x_seg, wavelet='db4', level=1)

        X = create_X_feature(X, x_seg, segment, use_wavelets)
        y = create_y(y, y_seg, segment)
        i += 1

    res = Parallel(n_jobs=6, verbose=2)(delayed(create_X_feature)(X, x_seg, segment, use_wavelets) for segment in range(segments))

    if save:
        X.to_csv(f"X_{version}.csv")
        y.to_csv(f"y_{version}.csv")

    print(overlapping_indexes)
    return X, y


def create_submission_set(X, submission, rows, version, use_wavelets, save=True):
    print("Starting creation of submissions set")
    X_sub = pd.DataFrame(columns=X.columns, dtype=np.float64, index=submission.index)

    for seg_id in tqdm(X_sub.index):
        seg = pd.read_csv('input/test/' + seg_id + '.csv')

        x_seg = pd.Series(seg['acoustic_data'].values[seg.shape[0] - rows:])

        X_sub = create_X_feature(X_sub, x_seg, seg_id, use_wavelets)

    if save:
        X_sub.to_csv(f"X_sub_{version}.csv")

    return X_sub


def create_X_feature(X, x_seg, segment, use_wavelets):
    '''
    create the training dataframe from the statistics calculated by calculate_statistics.
    wavelet decompositions is used and calculate statistics is applied on each decomposition
    '''

    x_seg = x_seg - np.mean(x_seg)
    if use_wavelets:
        list_coeff = pywt.wavedec(x_seg, 'db4')
        list_coeff.append(x_seg.values)
    else:
        list_coeff = []
        list_coeff.append(x_seg.values)
    l = len(list_coeff)
    i = 1
    for coeff in list_coeff:
        wavelet_stats_dict = calculate_statistics(coeff)

        for key in wavelet_stats_dict:
            X.loc[segment, key + f'_{i}'] = wavelet_stats_dict[key]

        i += 1

    '''
    feature_list = []

    wavelet_stats_dict = Parallel(n_jobs=6, verbose=2)(delayed(calculate_statistics)(coeff) for coeff in list_coeff)
    for r in wavelet_stats_dict:
        feature_list.append(r)

    print(feature_list)
    X = pd.DataFrame(feature_list)
    '''
    return X


def calculate_statistics(list_values):
    stats_dict = {}
    crossings = calculate_crossings(list_values)
    x_seg = pd.Series(list_values)

    percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    hann_windows = [50, 150, 1500, 15000]
    spans = [300, 3000, 30000, 50000]
    windows = [10, 50, 100, 500, 1000, 10000]
    borders = list(range(-4000, 4001, 1000))
    peaks = [10, 20, 50, 100]
    coefs = [1, 5, 10, 50, 100]
    # lags = [10, 100, 1000, 10000]
    autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]

    if x_seg.shape[0] > 10000:
        stats_dict['no_zero_crossings'] = crossings[0]
        stats_dict['no_mean_crossings'] = crossings[1]
        stats_dict['entropy'] = calculate_entropy(list_values)
        stats_dict['power'] = np.sum(list_values**2) / 4e6
        stats_dict['mean'] = np.nanmean(list_values)
        stats_dict['std'] = np.nanstd(list_values)
        stats_dict['var'] = np.nanvar(list_values)
        stats_dict['rms'] = np.nanmean(np.sqrt(list_values**2))
        stats_dict['mad'] = np.nanmean(np.absolute(list_values - np.nanmean(list_values)))
        stats_dict['kurt'] = kurtosis(list_values)
        stats_dict['sk'] = skew(list_values)
        stats_dict['max'] = np.max(list_values)
        stats_dict['min'] = np.min(list_values)

        # percentiles on original and absolute values
        for p in percentiles:
            stats_dict[f'percentile_{p}'] = np.nanpercentile(list_values, p)
            stats_dict[f'abs_percentile_{p}'] = np.nanpercentile(np.abs(list_values), p)

        # peaks
        for peak in peaks:
            stats_dict[f'num_peaks_{peak}'] = feature_calculators.number_peaks(list_values, peak)

         # geometric and harminic means
        stats_dict['hmean'] = hmean(np.abs(list_values[np.nonzero(list_values)[0]]))
        stats_dict['gmean'] = gmean(np.abs(list_values[np.nonzero(list_values)[0]]))

        # k-statistic and moments
        for i in range(1, 5):
            stats_dict[f'kstat_{i}'] = kstat(list_values, i)
            stats_dict[f'moment_{i}'] = moment(list_values, i)

        for i in [1, 2]:
            stats_dict[f'kstatvar_{i}'] = kstatvar(list_values, i)

        # autocorrelation
        for autocorr_lag in autocorr_lags:
            stats_dict[f'autocorrelation_{autocorr_lag}'] = feature_calculators.autocorrelation(list_values, autocorr_lag)
            stats_dict[f'c3_{autocorr_lag}'] = feature_calculators.c3(list_values, autocorr_lag)

        # stats_dict['trend'] = add_trend_feature(x_seg)
        # stats_dict['abs_trend'] = add_trend_feature(x_seg, abs_values=True)
        # stats_dict['abs_mean'] = np.abs(x_seg).mean()
        # stats_dict['abs_std'] = np.abs(x_seg).std()

        # stats_dict['Hilbert_mean'] = np.abs(hilbert(x_seg)).mean()

        # for hw in hann_windows:
        #   stats_dict[f'Hann_window_mean_{hw}'] = (convolve(x_seg, hann(150), mode='same') / sum(hann(150))).mean()

        # stats_dict['classic_sta_lta1_mean'] = classic_sta_lta(x_seg, 500, 10000).mean()
        # stats_dict['classic_sta_lta2_mean'] = classic_sta_lta(x_seg, 5000, 100000).mean()
        # stats_dict['classic_sta_lta3_mean'] = classic_sta_lta(x_seg, 3333, 6666).mean()
        # stats_dict['classic_sta_lta4_mean'] = classic_sta_lta(x_seg, 10000, 25000).mean()
        # stats_dict['classic_sta_lta5_mean'] = classic_sta_lta(x_seg, 50, 1000).mean()
        # stats_dict['classic_sta_lta6_mean'] = classic_sta_lta(x_seg, 100, 5000).mean()
        # stats_dict['classic_sta_lta7_mean'] = classic_sta_lta(x_seg, 333, 666).mean()
        # stats_dict['classic_sta_lta8_mean'] = classic_sta_lta(x_seg, 4000, 10000).mean()
        # # stats_dict['Moving_average_700_mean'] = x_seg.rolling(window=700).mean().mean(skipna=True)

        # Number of points in the range
        stats_dict['range_minf_m4000'] = feature_calculators.range_count(list_values, -np.inf, -4000)
        stats_dict['range_p4000_pinf'] = feature_calculators.range_count(list_values, 4000, np.inf)

        for i, j in zip(borders, borders[1:]):
            stats_dict[f'range_{i}_{j}'] = feature_calculators.range_count(list_values, i, j)

        list_values_50000 = list_values[0: 49999]
        crossings_50000 = calculate_crossings(list_values_50000)
        stats_dict['no_zero_crossings_50000'] = crossings_50000[0]
        stats_dict['no_mean_crossings_50000'] = crossings_50000[1]
        stats_dict['entropy_50000'] = calculate_entropy(list_values_50000)
        stats_dict['power_50000'] = np.sum(list_values_50000**2) / 4e6
        stats_dict['n5_50000'] = np.nanpercentile(list_values_50000, 5)
        stats_dict['n25_50000'] = np.nanpercentile(list_values_50000, 25)
        stats_dict['n75_50000'] = np.nanpercentile(list_values_50000, 75)
        stats_dict['n95_50000'] = np.nanpercentile(list_values_50000, 95)
        stats_dict['median_50000'] = np.nanpercentile(list_values_50000, 50)
        stats_dict['mean_50000'] = np.nanmean(list_values_50000)
        stats_dict['std_50000'] = np.nanstd(list_values_50000)
        stats_dict['var_50000'] = np.nanvar(list_values_50000)
        stats_dict['rms_50000'] = np.nanmean(np.sqrt(list_values_50000**2))
        stats_dict['mad_50000'] = np.nanmean(np.absolute(list_values_50000 - np.nanmean(list_values_50000)))
        stats_dict['kurt_50000'] = kurtosis(list_values_50000)
        stats_dict['sk_50000'] = skew(list_values_50000)
        stats_dict['max_50000'] = np.max(list_values_50000)
        stats_dict['min_50000'] = np.min(list_values_50000)
        stats_dict['num_peaks_50000_10'] = feature_calculators.number_peaks(list_values_50000, 10)

        list_values_100000 = list_values[50000: 999999]
        crossings_100000 = calculate_crossings(list_values_100000)
        stats_dict['no_zero_crossings_100000'] = crossings_100000[0]
        stats_dict['no_mean_crossings_100000'] = crossings_100000[1]
        stats_dict['entropy_100000'] = calculate_entropy(list_values_100000)
        stats_dict['power_100000'] = np.sum(list_values_100000**2) / 4e6
        stats_dict['n5_100000'] = np.nanpercentile(list_values_100000, 5)
        stats_dict['n25_100000'] = np.nanpercentile(list_values_100000, 25)
        stats_dict['n75_100000'] = np.nanpercentile(list_values_100000, 75)
        stats_dict['n95_100000'] = np.nanpercentile(list_values_100000, 95)
        stats_dict['median_100000'] = np.nanpercentile(list_values_100000, 50)
        stats_dict['mean_100000'] = np.nanmean(list_values_100000)
        stats_dict['std_100000'] = np.nanstd(list_values_100000)
        stats_dict['var_100000'] = np.nanvar(list_values_100000)
        stats_dict['rms_100000'] = np.nanmean(np.sqrt(list_values_100000**2))
        stats_dict['mad_100000'] = np.nanmean(np.absolute(list_values_100000 - np.nanmean(list_values_100000)))
        stats_dict['kurt_100000'] = kurtosis(list_values_100000)
        stats_dict['sk_100000'] = skew(list_values_100000)
        stats_dict['max_100000'] = np.max(list_values_100000)
        stats_dict['min_100000'] = np.min(list_values_100000)
        stats_dict['num_peaks_100000_10'] = feature_calculators.number_peaks(list_values_100000, 10)

        list_values_150000 = list_values[100000: 149999]
        crossings_150000 = calculate_crossings(list_values_150000)
        stats_dict['no_zero_crossings_150000'] = crossings_150000[0]
        stats_dict['no_mean_crossings_150000'] = crossings_150000[1]
        stats_dict['entropy_150000'] = calculate_entropy(list_values_150000)
        stats_dict['power_150000'] = np.sum(list_values_150000**2) / 4e6
        stats_dict['n5_150000'] = np.nanpercentile(list_values_150000, 5)
        stats_dict['n25_150000'] = np.nanpercentile(list_values_150000, 25)
        stats_dict['n75_150000'] = np.nanpercentile(list_values_150000, 75)
        stats_dict['n95_150000'] = np.nanpercentile(list_values_150000, 95)
        stats_dict['median_150000'] = np.nanpercentile(list_values_150000, 50)
        stats_dict['mean_150000'] = np.nanmean(list_values_150000)
        stats_dict['std_150000'] = np.nanstd(list_values_150000)
        stats_dict['var_150000'] = np.nanvar(list_values_150000)
        stats_dict['rms_150000'] = np.nanmean(np.sqrt(list_values_150000**2))
        stats_dict['mad_150000'] = np.nanmean(np.absolute(list_values_150000 - np.nanmean(list_values_150000)))
        stats_dict['kurt_150000'] = kurtosis(list_values_150000)
        stats_dict['sk_150000'] = skew(list_values_150000)
        stats_dict['max_150000'] = np.max(list_values_150000)
        stats_dict['min_150000'] = np.min(list_values_150000)
        stats_dict['num_peaks_100000_10'] = feature_calculators.number_peaks(list_values_150000, 10)

        for window in windows:
            if x_seg.shape[0] > window:
                x_roll_std = x_seg.rolling(window).std().dropna().values
                x_roll_mean = x_seg.rolling(window).mean().dropna().values

                stats_dict['ave_roll_std_' + str(window)] = x_roll_std.mean()
                stats_dict['std_roll_std_' + str(window)] = x_roll_std.std()
                stats_dict['max_roll_std_' + str(window)] = x_roll_std.max()
                stats_dict['min_roll_std_' + str(window)] = x_roll_std.min()
                stats_dict['q01_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.01)
                stats_dict['q05_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.05)
                stats_dict['q95_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.95)
                stats_dict['q99_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.99)
                stats_dict['av_change_abs_roll_std_' + str(window)] = np.mean(np.diff(x_roll_std))
                stats_dict['av_change_rate_roll_std_' + str(window)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                stats_dict['abs_max_roll_std_' + str(window)] = np.abs(x_roll_std).max()
                stats_dict['ave_roll_mean_' + str(window)] = x_roll_mean.mean()
                stats_dict['std_roll_mean_' + str(window)] = x_roll_mean.std()
                stats_dict['max_roll_mean_' + str(window)] = x_roll_mean.max()
                stats_dict['min_roll_mean_' + str(window)] = x_roll_mean.min()
                stats_dict['q01_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.01)
                stats_dict['q05_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.05)
                stats_dict['q95_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.95)
                stats_dict['q99_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.99)
                stats_dict['av_change_abs_roll_mean_' + str(window)] = np.mean(np.diff(x_roll_mean))
                with np.errstate(all='ignore'):
                    stats_dict['av_change_rate_roll_mean_' + str(window)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                stats_dict['abs_max_roll_mean_' + str(window)] = np.abs(x_roll_mean).max()

    # stats_dict['mean_change_abs'] = np.mean(np.diff(x_seg))
    # stats_dict['mean_change_rate'] = calc_change_rate(x_seg)
    # stats_dict['abs_max'] = np.abs(x_seg).max()
    # stats_dict['abs_min'] = np.abs(x_seg).min()
    # stats_dict['std_first_100000'] = x_seg[:50000].std()
    # stats_dict['std_last_50000'] = x_seg[-50000:].std()
    # stats_dict['std_first_10000'] = x_seg[:10000].std()
    # stats_dict['std_last_10000'] = x_seg[-10000:].std()
    # stats_dict['ave_first_50000'] = x_seg[:50000].mean()
    # stats_dict['ave_last_50000'] = x_seg[-50000:].mean()
    # stats_dict['ave_first_10000'] = x_seg[:10000].mean()
    # stats_dict['ave_last_10000'] = x_seg[-10000:].mean()
    # stats_dict['min_first_50000'] = x_seg[:50000].min()
    # stats_dict['min_last_50000'] = x_seg[-50000:].min()
    # stats_dict['min_first_10000'] = x_seg[:10000].min()
    # stats_dict['min_last_10000'] = x_seg[-10000:].min()
    # stats_dict['max_first_50000'] = x_seg[:50000].max()
    # stats_dict['max_last_50000'] = x_seg[-50000:].max()
    # stats_dict['max_first_10000'] = x_seg[:10000].max()
    # stats_dict['max_last_10000'] = x_seg[-10000:].max()
    # stats_dict['skew_first_50000'] = skew(x_seg[:50000])
    # stats_dict['skew_last_50000'] = skew(x_seg[-50000:])
    # stats_dict['skew_first_10000'] = skew(x_seg[:10000])
    # stats_dict['skew_last_10000'] = skew(x_seg[-10000:])
    # stats_dict['kurt_first_50000'] = kurtosis(x_seg[:50000])
    # stats_dict['kurt_last_50000'] = kurtosis(x_seg[-50000:])
    # stats_dict['kurt_first_10000'] = kurtosis(x_seg[:10000])
    # stats_dict['kurt_last_10000'] = kurtosis(x_seg[-10000:])
    # stats_dict['max_to_min'] = x_seg.max() / np.abs(x_seg.min())
    # stats_dict['max_to_min_diff'] = x_seg.max() - np.abs(x_seg.min())
    # stats_dict['count_big'] = len(x_seg[np.abs(x_seg) > 500])
    # stats_dict['sum'] = x_seg.sum()
    # stats_dict['mean_change_rate_first_50000'] = calc_change_rate(x_seg[:50000])
    # stats_dict['mean_change_rate_last_50000'] = calc_change_rate(x_seg[-50000:])
    # stats_dict['mean_change_rate_first_10000'] = calc_change_rate(x_seg[:10000])
    # stats_dict['mean_change_rate_last_10000'] = calc_change_rate(x_seg[-10000:])
    # stats_dict['q95'] = np.quantile(x_seg, 0.95)
    # stats_dict['q99'] = np.quantile(x_seg, 0.99)
    # stats_dict['q05'] = np.quantile(x_seg, 0.05)
    # stats_dict['q01'] = np.quantile(x_seg, 0.01)
    # stats_dict['abs_q95'] = np.quantile(np.abs(x_seg), 0.95)
    # stats_dict['abs_q99'] = np.quantile(np.abs(x_seg), 0.99)
    # stats_dict['abs_q05'] = np.quantile(np.abs(x_seg), 0.05)
    # stats_dict['abs_q01'] = np.quantile(np.abs(x_seg), 0.01)

    # ewma = pd.Series.ewm
    # stats_dict['exp_Moving_average_300_mean'] = (ewma(x_seg, span=300).mean()).mean(skipna=True)
    # stats_dict['exp_Moving_average_3000_mean'] = ewma(x_seg, span=3000).mean().mean(skipna=True)
    # stats_dict['exp_Moving_average_30000_mean'] = ewma(x_seg, span=30000).mean().mean(skipna=True)

    # no_of_std = 3
    # stats_dict['MA_700MA_std_mean'] = x_seg.rolling(window=700).std().mean()
    # stats_dict['MA_700MA_BB_high_mean'] = (X.loc[segment, 'Moving_average_700_mean'] + no_of_std * X.loc[segment, 'MA_700MA_std_mean']).mean()
    # stats_dict['MA_700MA_BB_low_mean'] = (X.loc[segment, 'Moving_average_700_mean'] - no_of_std * X.loc[segment, 'MA_700MA_std_mean']).mean()
    # stats_dict['MA_400MA_std_mean'] = x_seg.rolling(window=400).std().mean()
    # stats_dict['MA_400MA_BB_high_mean'] = (X.loc[segment, 'Moving_average_700_mean'] + no_of_std * X.loc[segment, 'MA_400MA_std_mean']).mean()
    # stats_dict['MA_400MA_BB_low_mean'] = (X.loc[segment, 'Moving_average_700_mean'] - no_of_std * X.loc[segment, 'MA_400MA_std_mean']).mean()
    # stats_dict['MA_1000MA_std_mean'] = x_seg.rolling(window=1000).std().mean()

    # stats_dict['iqr'] = np.subtract(*np.percentile(x_seg, [75, 25]))
    # stats_dict['q999'] = np.quantile(x_seg, 0.999)
    # stats_dict['q001'] = np.quantile(x_seg, 0.001)
    # stats_dict['ave10'] = trim_mean(x_seg, 0.1)

    return stats_dict


# =====================================================================================
# Filtering signal
# =====================================================================================
from scipy.signal import butter, deconvolve
#SAMPLE_RATE = 4000


def get_filter_coef(btype, cutoff=10000, low=10000, high=50000, sample_rate=150000):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """

    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * sample_rate 

    # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
    if btype == 'lowpass':
        b, a = butter(4, Wn=cutoff / nyquist, btype='lowpass')
    elif btype == 'highpass':
        b, a = butter(4, Wn=cutoff / nyquist, btype='highpass')
    elif btype == 'bandpass':
        b, a = butter(4, Wn=(low / nyquist, high / nyquist), btype='bandpass')

    return b, a


def high_pass_filter(x, low_cutoff=10000, SAMPLE_RATE=4000000):
    """
    From @randxie https://github.com/randxie/Kaggle-VSB-Baseline/blob/master/src/utils/util_signal.py
    Modified to work with scipy version 1.1.0 which does not have the fs parameter
    """

    # nyquist frequency is half the sample rate https://en.wikipedia.org/wiki/Nyquist_frequency
    nyquist = 0.5 * SAMPLE_RATE
    norm_low_cutoff = low_cutoff / nyquist

    sos = butter(4, Wn=norm_low_cutoff, btype='highpass', output='sos')
    filtered_sig = sosfilt(sos, x)

    return filtered_sig

def denoise_signal(x, wavelet='db4', level=1):
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf

    Rhe filtered signals are now passed through wavelet decomposition (and the wavelet coefficients are obtained). 
    It has to be understood that the raw signal we are working with is a convolution of the artificial and real impulse signals and this is why we need to "wavelet decomposition". 
    The process is a sort of "deconvolution", which means it undoes the convolution process and uncovers the real Earth impulse from the mixed seismogram and the artificial impulse. 
    e make use of the MAD value to understand the randomness in the signal and accordingly decide the minimum threshold for the wavelet coefficients in the time series. 
    We filter out the low coefficients from the wavelet coefficients and reconstruct the real Earth signal from the remaining coefficients and that's it; 
    we have successfully removed the impulse signal from the seismogram and obtained the real Earth signal.
    """

    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation
    sigma = (1 / 0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

# =====================================================================================
# Functions to calculate features
# =====================================================================================


def maddest(d, axis=None):
    """
    This calculates the mean of the absolute values of the deviations of the individual numbers in the time series from the mean of the time series. 
    It is a measure of entropy or disorder in the time series. The greater the MAD value, the more disorderly and unpredictable the time series is.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def calculate_entropy(list_values):
    list_values = list_values[~np.isinf(list_values)]
    list_values = list_values[~np.isnan(list_values)]
    counter_values = Counter([round(x) for x in list_values]).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    ent = entropy(probabilities)
    return ent


def add_trend_feature(arr, abs_values=False):
    ''' Returns the trend coefficient of an array obtained from a linear regression'''
    index = np.array(range(len(arr))).reshape(-1, 1)
    if abs_values:
        arr = np.abs(arr)
    linreg = LinearRegression()
    linreg.fit(index, arr)
    return linreg.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    '''Anti-triggering algorithm STA/LTA (Short Time Average over Long Time Average)
    '''
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float64)

    # Copy for lta
    lta = sta.copy()

    # Compute the sta and lta
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


def calc_change_rate(x):
    # Différences à l'ordre 1, divisées par les n-1 premiers termes : (x_1 - x_0) / x_0
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]  # np.nonzero returns indeces of the element that are not zero
    change = change[~np.isnan(change)]  # return indices of change that are unmbers
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[1: N // 2 + 1])
    return f_values, fft_values


def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]


def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values


def get_first_n_peaks(x, y, no_peaks=5):
    # Select the first no_peaks elements of x and y
    x_, y_ = list(x), list(y)
    if len(x_) != 0:
        y_, x_ = (list(t) for t in zip(*sorted(zip(y_, x_), reverse=True)))
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0] * missing_no_peaks, y_ + [0] * missing_no_peaks


def get_peaks_coordinates(x_values, y_values, mph, reorder=True):
    # Get the indexes of peaks, peaks are values above mph
    indices_peaks = detect_peaks(y_values, mph=mph)
    # Get the x and y coordinates of the peaks and select the first no_peaks of them
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    # reorder peaks according to their amplitude
    if reorder:
        peaks_y, peaks_x = (list(t) for t in zip(*sorted(zip(peaks_y, peaks_x))))
    return peaks_x, peaks_y


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's 
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def create_X_feature_old(X, x_seg, segment):

    # X.loc[segment, 'mean'] = x_seg.mean()
    # X.loc[segment, 'std'] = x_seg.std()
    X.loc[segment, 'max'] = x_seg.max()
    X.loc[segment, 'min'] = x_seg.min()

    # FFT transform values
    zc = np.fft.fft(x_seg)
    N = len(x_seg)  # samples
    f_s = 4e6  # recording frequency
    t_n = N / f_s  # recording time (sec), N/f_s
    T = t_n / N

    percentile = 5
    signal_min = np.nanpercentile(x_seg, percentile)
    signal_max = np.nanpercentile(x_seg, 100 - percentile)
    # ijk = (100 - 2*percentile)/10
    denominator = 10
    mph = signal_min + (signal_max - signal_min) / denominator
    fft_peaks_x_array, fft_peaks_y_array = get_peaks_coordinates(*get_fft_values(x_seg, T, N, f_s), mph)
    psd_peaks_x_array, psd_peaks_y_array = get_peaks_coordinates(*get_psd_values(x_seg, T, N, f_s), mph)

    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    X.loc[segment, 'fft_peak_x_1'] = fft_peaks_x_array[0]
    X.loc[segment, 'fft_peak_x_2'] = fft_peaks_x_array[1]
    X.loc[segment, 'fft_peak_x_3'] = fft_peaks_x_array[2]
    X.loc[segment, 'fft_peak_x_4'] = fft_peaks_x_array[3]
    X.loc[segment, 'fft_peak_x_5'] = fft_peaks_x_array[4]
    X.loc[segment, 'fft_peak_y_1'] = fft_peaks_y_array[0]
    X.loc[segment, 'fft_peak_y_2'] = fft_peaks_y_array[1]
    X.loc[segment, 'fft_peak_y_3'] = fft_peaks_y_array[2]
    X.loc[segment, 'fft_peak_y_4'] = fft_peaks_y_array[3]
    X.loc[segment, 'fft_peak_y_5'] = fft_peaks_y_array[4]
    X.loc[segment, 'psd_peak_x_1'] = psd_peaks_x_array[0]
    X.loc[segment, 'psd_peak_x_2'] = psd_peaks_x_array[1]
    X.loc[segment, 'psd_peak_x_3'] = psd_peaks_x_array[2]
    X.loc[segment, 'psd_peak_x_4'] = psd_peaks_x_array[3]
    X.loc[segment, 'psd_peak_x_5'] = psd_peaks_x_array[4]
    X.loc[segment, 'psd_peak_y_1'] = psd_peaks_y_array[0]
    X.loc[segment, 'psd_peak_y_2'] = psd_peaks_y_array[1]
    X.loc[segment, 'psd_peak_y_3'] = psd_peaks_y_array[2]
    X.loc[segment, 'psd_peak_y_4'] = psd_peaks_y_array[3]
    X.loc[segment, 'psd_peak_y_5'] = psd_peaks_y_array[4]
    X.loc[segment, 'Rstd'] = realFFT.std()
    X.loc[segment, 'Rmax'] = realFFT.max()
    X.loc[segment, 'Rmin'] = realFFT.min()
    X.loc[segment, 'Rmean'] = realFFT.mean()
    X.loc[segment, 'Imean'] = imagFFT.mean()
    X.loc[segment, 'Istd'] = imagFFT.std()
    X.loc[segment, 'Imax'] = imagFFT.max()
    X.loc[segment, 'Imin'] = imagFFT.min()
    X.loc[segment, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    X.loc[segment, 'Rstd__last_5000'] = realFFT[-5000:].std()
    X.loc[segment, 'Rmax_last_5000'] = realFFT[-5000:].max()
    X.loc[segment, 'Rmin_last_5000'] = realFFT[-5000:].min()
    X.loc[segment, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    X.loc[segment, 'Rstd_last_15000'] = realFFT[-15000:].std()
    X.loc[segment, 'Rmax_last_15000'] = realFFT[-15000:].max()
    X.loc[segment, 'Rmin_last_15000'] = realFFT[-15000:].min()

    # wavelets

    list_coeff = pywt.wavedec(x_seg, 'db4')
    list_coeff.append(x_seg.values)
    wavelet_features = []
    l = len(list_coeff)
    for coeff in list_coeff:
        wavelet_features += get_features(coeff)
    for i in range(0, l):
        X.loc[segment, f'wt_entropy_{i}'] = wavelet_features[l * i + 0]
        X.loc[segment, f'wt_no_zero_crossings_{i}'] = wavelet_features[l * i + 1]
        X.loc[segment, f'wt_no_mean_crossings_{i}'] = wavelet_features[l * i + 2]
        X.loc[segment, f'wt_n5_{i}'] = wavelet_features[l * i + 3]
        X.loc[segment, f'wt_n25_{i}'] = wavelet_features[l * i + 4]
        X.loc[segment, f'wt_n75_{i}'] = wavelet_features[l * i + 5]
        X.loc[segment, f'wt_n95_{i}'] = wavelet_features[l * i + 6]
        X.loc[segment, f'wt_median_{i}'] = wavelet_features[l * i + 7]
        X.loc[segment, f'wt_mean_{i}'] = wavelet_features[l * i + 8]
        X.loc[segment, f'wt_std_{i}'] = wavelet_features[l * i + 9]
        X.loc[segment, f'wt_var_{i}'] = wavelet_features[l * i + 10]
        X.loc[segment, f'wt_rms_{i}'] = wavelet_features[l * i + 11]
        X.loc[segment, f'wt_mad_{i}'] = wavelet_features[l * i + 12]
        X.loc[segment, f'wt_kurt_{i}'] = wavelet_features[l * i + 13]
        X.loc[segment, f'wt_sk_{i}'] = wavelet_features[l * i + 14]
        X.loc[segment, f'wt_en_{i}'] = wavelet_features[l * i + 15]

    X.loc[segment, 'mad'] = x_seg.mad()
    X.loc[segment, 'kurt'] = kurtosis(x_seg)
    X.loc[segment, 'skew'] = skew(x_seg)
    X.loc[segment, 'med'] = np.median(x_seg)

    X.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x_seg))
    X.loc[segment, 'mean_change_rate'] = calc_change_rate(x_seg)
    X.loc[segment, 'abs_max'] = np.abs(x_seg).max()
    X.loc[segment, 'abs_min'] = np.abs(x_seg).min()

    X.loc[segment, 'std_first_50000'] = x_seg[:50000].std()
    X.loc[segment, 'std_last_50000'] = x_seg[-50000:].std()
    X.loc[segment, 'std_first_10000'] = x_seg[:10000].std()
    X.loc[segment, 'std_last_10000'] = x_seg[-10000:].std()

    X.loc[segment, 'ave_first_50000'] = x_seg[:50000].mean()
    X.loc[segment, 'ave_last_50000'] = x_seg[-50000:].mean()
    X.loc[segment, 'ave_first_10000'] = x_seg[:10000].mean()
    X.loc[segment, 'ave_last_10000'] = x_seg[-10000:].mean()

    X.loc[segment, 'min_first_50000'] = x_seg[:50000].min()
    X.loc[segment, 'min_last_50000'] = x_seg[-50000:].min()
    X.loc[segment, 'min_first_10000'] = x_seg[:10000].min()
    X.loc[segment, 'min_last_10000'] = x_seg[-10000:].min()

    X.loc[segment, 'max_first_50000'] = x_seg[:50000].max()
    X.loc[segment, 'max_last_50000'] = x_seg[-50000:].max()
    X.loc[segment, 'max_first_10000'] = x_seg[:10000].max()
    X.loc[segment, 'max_last_10000'] = x_seg[-10000:].max()

    X.loc[segment, 'skew_first_50000'] = skew(x_seg[:50000])
    X.loc[segment, 'skew_last_50000'] = skew(x_seg[-50000:])
    X.loc[segment, 'skew_first_10000'] = skew(x_seg[:10000])
    X.loc[segment, 'skew_last_10000'] = skew(x_seg[-10000:])

    X.loc[segment, 'kurt_first_50000'] = kurtosis(x_seg[:50000])
    X.loc[segment, 'kurt_last_50000'] = kurtosis(x_seg[-50000:])
    X.loc[segment, 'kurt_first_10000'] = kurtosis(x_seg[:10000])
    X.loc[segment, 'kurt_last_10000'] = kurtosis(x_seg[-10000:])

    X.loc[segment, 'max_to_min'] = x_seg.max() / np.abs(x_seg.min())
    X.loc[segment, 'max_to_min_diff'] = x_seg.max() - np.abs(x_seg.min())
    X.loc[segment, 'count_big'] = len(x_seg[np.abs(x_seg) > 500])
    X.loc[segment, 'sum'] = x_seg.sum()

    X.loc[segment, 'mean_change_rate_first_50000'] = calc_change_rate(x_seg[:50000])
    X.loc[segment, 'mean_change_rate_last_50000'] = calc_change_rate(x_seg[-50000:])
    X.loc[segment, 'mean_change_rate_first_10000'] = calc_change_rate(x_seg[:10000])
    X.loc[segment, 'mean_change_rate_last_10000'] = calc_change_rate(x_seg[-10000:])

    X.loc[segment, 'q95'] = np.quantile(x_seg, 0.95)
    X.loc[segment, 'q99'] = np.quantile(x_seg, 0.99)
    X.loc[segment, 'q05'] = np.quantile(x_seg, 0.05)
    X.loc[segment, 'q01'] = np.quantile(x_seg, 0.01)

    X.loc[segment, 'abs_q95'] = np.quantile(np.abs(x_seg), 0.95)
    X.loc[segment, 'abs_q99'] = np.quantile(np.abs(x_seg), 0.99)
    X.loc[segment, 'abs_q05'] = np.quantile(np.abs(x_seg), 0.05)
    X.loc[segment, 'abs_q01'] = np.quantile(np.abs(x_seg), 0.01)

    X.loc[segment, 'trend'] = add_trend_feature(x_seg)
    X.loc[segment, 'abs_trend'] = add_trend_feature(x_seg, abs_values=True)
    X.loc[segment, 'abs_mean'] = np.abs(x_seg).mean()
    X.loc[segment, 'abs_std'] = np.abs(x_seg).std()

    X.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x_seg)).mean()
    X.loc[segment, 'Hann_window_mean'] = (convolve(x_seg, hann(150), mode='same') / sum(hann(150))).mean()
    X.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x_seg, 500, 10000).mean()
    X.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x_seg, 5000, 100000).mean()
    X.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x_seg, 3333, 6666).mean()
    X.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x_seg, 10000, 25000).mean()
    X.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x_seg, 50, 1000).mean()
    X.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x_seg, 100, 5000).mean()
    X.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x_seg, 333, 666).mean()
    X.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x_seg, 4000, 10000).mean()
    X.loc[segment, 'Moving_average_700_mean'] = x_seg.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm
    X.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x_seg, span=300).mean()).mean(skipna=True)
    X.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x_seg, span=3000).mean().mean(skipna=True)
    X.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x_seg, span=30000).mean().mean(skipna=True)

    no_of_std = 3
    X.loc[segment, 'MA_700MA_std_mean'] = x_seg.rolling(window=700).std().mean()
    X.loc[segment, 'MA_700MA_BB_high_mean'] = (X.loc[segment, 'Moving_average_700_mean'] + no_of_std * X.loc[segment, 'MA_700MA_std_mean']).mean()
    X.loc[segment, 'MA_700MA_BB_low_mean'] = (X.loc[segment, 'Moving_average_700_mean'] - no_of_std * X.loc[segment, 'MA_700MA_std_mean']).mean()
    X.loc[segment, 'MA_400MA_std_mean'] = x_seg.rolling(window=400).std().mean()
    X.loc[segment, 'MA_400MA_BB_high_mean'] = (X.loc[segment, 'Moving_average_700_mean'] + no_of_std * X.loc[segment, 'MA_400MA_std_mean']).mean()
    X.loc[segment, 'MA_400MA_BB_low_mean'] = (X.loc[segment, 'Moving_average_700_mean'] - no_of_std * X.loc[segment, 'MA_400MA_std_mean']).mean()
    X.loc[segment, 'MA_1000MA_std_mean'] = x_seg.rolling(window=1000).std().mean()
    X.drop('Moving_average_700_mean', axis=1, inplace=True)

    X.loc[segment, 'iqr'] = np.subtract(*np.percentile(x_seg, [75, 25]))
    X.loc[segment, 'q999'] = np.quantile(x_seg, 0.999)
    X.loc[segment, 'q001'] = np.quantile(x_seg, 0.001)
    X.loc[segment, 'ave10'] = trim_mean(x_seg, 0.1)

    for windows in [10, 100, 1000, 10000]:
        x_roll_std = x_seg.rolling(windows).std().dropna().values
        x_roll_mean = x_seg.rolling(windows).mean().dropna().values

        X.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        with np.errstate(all='ignore'):
            X.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    return X
