import pandas as pd
import numpy as np
from itertools import combinations
import logging

from scipy.stats import randint, uniform
from scipy.optimize import minimize

from sklearn.utils import shuffle
from sklearn.model_selection import KFold, RandomizedSearchCV, ParameterSampler
from sklearn.externals import joblib

from mlens.ensemble import SuperLearner
from mlens.metrics import make_scorer

from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def stacking(clf_array, X_test, y_test):

    for clf in clf_array:
        y_pred = clf.predict(X_test)
        train_stack = np.vstack(train_stack, y_pred)
    train_stack = train_stack.transpose()
    train_stack = pd.DataFrame(train_stack, columns=['rf', 'lgb', 'xgb', 'svr', 'kr'])
    test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_svr, prediction_svr1, prediction_r, prediction_cat]).transpose()


def simple_blending(X, y, X_sub, pipe_array=None, from_file=True):

    print("Computing blended predictions...")

    final_prediction = np.zeros(len(X_sub))

    if from_file:

        files = ['submission_bag_1554462105.4202418.csv',
                 'submission_bag_1554530036.1343007.csv',
                 'submission_bag_2_212_1555544962.0909512.csv',
                 #'submission_lgb_1554423662.9597363.csv',
                 'submission_rf_1555144022.8721962.csv',
                 'submission_rf_1554425491.9242666.csv',
                 'submission_lgb_4_212_1555314583.8839114.csv',
                 'submission_lgb_1554335418.928878.csv',


                 'submission_oof_rf_8_2.036_212_1557637679.csv', #1.449
                 'submission_oof_bag_8_2.03_212_1557559086.csv', #1.449
                 'submission_oof_lgb_8_2.249_212_1557361721.csv', #1.468
                 #'submission_oof_lgb_8_2.051_212_1557358091.csv', #1.479
                 #'submission_oof_lgb_8_2.144_212_1557355563.csv', #1.496
                 #'submission_oof_lgb_8_1.996_212_1557316941.csv', #1.497
                 #'submission_oof_lgb_8_1.996_212_1557316941.csv', #1.497 
                 #'submission_oof_xgb_8_2.061_212_1557213920.csv', #1.480
                 #'submission_oof_lgb_8_2.03_212_1557183418.csv', #1.490
                 'submission_oof_lgb_8_2.057_212_1557180427.csv', #1.439
                 'andrew_blending.csv',
                 'andrew_gpsubmission.csv'

                 'submission_oof_lgb_8_0.394_212_1558435062.csv' #1.440, ttf, 
                 ]

        for file in files:
            predictions = pd.read_csv('output/' + file)
            final_prediction += predictions['time_to_failure'].values

        result = final_prediction / len(files)
    else:

        print(len(pipe_array))
        for clf in pipe_array:
            print(clf)
            predictions = clf.predict(X_test)
            print("prediction", predictions)
            final_prediction += predictions
            print("final prediction", final_prediction)

        result = final_prediction / len(pipe_array)

    return result, len(files)


def get_model(job_load_name):

    clf = joblib.load('output/' + job_load_name + '.pkl')

    return clf


# ================================================
# BLENDING
# ================================================

class blending(object):

  def __init__(self, oof_dict, cv, quake_cat_array, score_func, **kwargs):
    self.oof_dict = oof_dict
    self.cv = cv
    self.quake_cat_array = quake_cat_array
    self.score_func = score_func

    self.best_score_ = None
    self.best_weights_ = None

  def fit(self, y_train, n_iter):

    '''
    Find the best weights to blend the results of several models given their out-of-fold predictions
    '''
    oof_predictions = []
    for model_name in self.oof_dict:
      oof_predictions.append(np.array(self.oof_dict[model_name]['oof_predictions']))
    oof_predictions = np.array(oof_predictions)

    # for train_index, val_index in cv.split():
    #    X_train, X_val = oof_predictions[train_index], oof_predictions[val_index]
    #    y_train, y_val = y[train_index], y[val_index]

    results_list = []
    weights_list = []
    
    y = []
    for train_indexes, val_indexes in self.cv.split(y_train, self.quake_cat_array):
      y_val = y_train[val_indexes]
      y += list(y_val)


    for i in range(n_iter):
      starting_values = np.random.uniform(size=len(oof_predictions))

      bounds = [(0, 1)] * len(oof_predictions)

      results = minimize(self.mae_min_func,
                         starting_values,
                         args=(oof_predictions, y),
                         method='L-BFGS-B',
                         bounds=bounds,
                         options={'disp': False,
                                  'maxiter': 100000})

      result = results['fun']
      weight = results['x']
      results_list.append(result)
      weights_list.append(weight)

      print(f'{i+1}\tScore: {result}\tWeights: {weight}')

    self.best_score_ = np.min(results_list)
    self.best_weights_ = weights_list[np.argmin(results_list)]

    print(f'\n Ensemble Score: {self.best_score_}')
    print(f'\n Best Weights: {self.best_weights_}')

    return self

  def predict(self, X_sub, submission, **kwargs):

    final_weighted_predictions = np.zeros(X_sub.shape[0])

    '''
    if refit:
      try:
        X, y = kwarg['X'], kwargs['y']
      except:
        print('refit = True but X and y parameters are not given')
        sys.exit()
    '''
    k = 0
    for model_name in self.oof_dict:

      model = self.oof_dict[model_name]['best_estimator']
      bs = self.oof_dict[model_name]['best_score_mean']
      y_sub_pred = model.predict(X_sub)
      #submission['time_to_failure'] = y_sub_pred
      # submission.to_csv(f'output/submission_{model_name}_{VERSION}_{bs}_{RANDOM_SEED}_{TIME}.csv')
      final_weighted_predictions += y_sub_pred * self.best_weights_[k]
      k += 1

    return final_weighted_predictions


  def mae_min_func(self, weights, oof_predictions, y_train):
    '''
    Calculates the score obtained with the weighted predictions
    '''
    final_prediction = 0

    for weight, prediction in zip(weights, oof_predictions):
        final_prediction += weight * prediction

    return self.score_func(final_prediction, y_train)





def fit_blending(oof_predictions, y_train, cv, score_func):


    oof_predictions = np.array(oof_predictions)

    # for train_index, val_index in cv.split():
    #    X_train, X_val = oof_predictions[train_index], oof_predictions[val_index]
    #    y_train, y_val = y[train_index], y[val_index]

    results_list = []
    weights_list = []

    for i in range(1000):
        starting_values = np.random.uniform(size=len(oof_predictions))

        bounds = [(0, 1)] * len(oof_predictions)

        results = minimize(mae_min_func,
                           starting_values,
                           args=(oof_predictions, y_train, score_func),
                           method='L-BFGS-B',
                           bounds=bounds,
                           options={'disp': False,
                                    'maxiter': 100000})

        result = results['fun']
        weight = results['x']
        results_list.append(result)
        weights_list.append(weight)

        print(f'{i+1}\tScore: {result}\tWeights: {weight}')

    best_score = np.min(results_list)
    best_weights = weights_list[np.argmin(results_list)]

    print(f'\n Ensemble Score: {best_score}')
    print(f'\n Best Weights: {best_weights}')

    return best_score, best_weights



'''
def get_clf_array(final=False):

    clf_names = ['lgb', 'Random Forest', 'xgb', 'svr', 'kr']

    # if final is True we get the models fitted on the full dataset, else the gridsearch results
    if final:
        clf_rf = joblib.load('output/clf_bag_final_1554462103.1869643.pkl')
        clf_lgb = joblib.load('output/clf_bag_final_2_212_1555544960.7086468.pkl')
        clf_xgb = joblib.load('output/clf_bag_final_1554530034.4926913.pkl')
        clf_bag = joblib.load('output/clf_rf_final_1555141695.7449248.pkl')
        clf_bag = joblib.load('output/clf_rf_final_1554421248.4070334.pkl')
        clf_bag = joblib.load('output/clf_lgb_final_4_212_1555314583.6455483.pkl')
    else:
        clf_rf = joblib.load('output/clf_rf_1553600814.9808807.pkl')
        clf_lgb = joblib.load('output/clf_lgb_1553835406.6128614.pkl')
        clf_xgb = joblib.load('output/clf_xgb_1553678592.43716.pkl')
        clf_svr = joblib.load('output/clf_svr_1553842774.3089724.pkl')
        clf_kr = joblib.load('output/clf_kr_1553762407.657696.pkl')
        clf_bag = joblib.load('output/clf_bag_final_1554530034.4926913.pkl')

    clf_array = [clf_lgb.best_estimator_.named_steps['lgb'],
                 clf_rf.best_estimator_.named_steps['rf'],
                 clf_xgb.best_estimator_.named_steps['xgb'],
                 clf_svr.best_estimator_.named_steps['svr']]
    pipe_array = [clf_lgb.best_estimator_,
                  clf_rf.best_estimator_,
                  clf_xgb.best_estimator_,
                  clf_svr.best_estimator_]

    return clf_names, clf_array, pipe_array
'''
# =========================================

'''
def get_oof_stuff(clf, X, y, cv, score_func):
'''
    #Computes the oof predictions of the best model
'''
    best_model = clf.best_estimator_
    X = X.values
    oof_predictions = []
    oof_scores = []

    for train_index, val_index in cv.split():
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        best_model.fit(X_train, y_train)
        y_val_predictions = best_model.predict(X_val).tolist()

        oof_predictions += y_val_predictions
        oof_scores.append(score_func(y_val_predictions, y_val))

    return oof_scores, np.array(oof_predictions)
'''

class CustomRandomizedSearchCV(object):
    def __init__(self, pipeline, param_distributions, metric, higher_is_better, cv, quake_cat_array, refit, n_iter, n_jobs, random_state, **kwargs):
        self.pipeline = pipeline
        self.param_distributions = param_distributions
        self.metric = metric
        if higher_is_better: self.i = 1
        else: self.i = -1
        self.cv = cv
        self.quake_cat_array = np.array(quake_cat_array)
        self.refit = refit
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.best_score_ = None
        self.best_scores_per_fold_ = None
        self.oof_predictions_ = None
        self.best_oof_scores_ = None
        self.best_params_ = None 
        self.best_estimator_ = None
        self.y_fold_ = None
        self.oof_score_ = None

    def fit(self, X, y, **kwargs):

        param_list = ParameterSampler(self.param_distributions, n_iter=self.n_iter)

        #fold_score, parameters, oof_predictions = fit_regressor(X, y, parameters)
        # for parameters in param_list:
        #   _fit_regressor(self.pipeline, X, y, self.quake_cat_array, parameters, self.metric, self.cv, self.random_state)
        print(f"Fitting piepline using {self.n_iter} sets of parameters over {self.cv.get_n_splits()} folds")
        param_scores = Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(_fit_regressor)(self.pipeline, X, y, self.quake_cat_array, parameters, self.metric, self.cv, self.random_state) for parameters in param_list)
        self.best_score_, self.best_scores_per_fold_, self.best_estimator_, self.best_params_, self.oof_predictions_, self.y_fold_, self.oof_score_ = max(param_scores, key=lambda x: self.i * x[0])

        print(f'Best scoring param is {self.best_params_} with score {self.best_score_}.')

        if isinstance(self.refit, str):
            self.pipeline.set_params(**self.best_params_)
            self.pipeline.fit(X, y)

        return self

    def predict(self, X_sub,  use_oof_estimate=None, X=None, y=None, cv=None, quake_cat_array=None, **kwargs):

        y_sub_pred = np.zeros(len(X_sub))
        if use_oof_estimate:
          for train_indexes, val_indexes in cv.split(X, quake_cat_array):
            X_train = X.iloc[train_indexes]
            y_train  = y[train_indexes]
            quake_cat_array_train = np.array(quake_cat_array)[train_indexes]

            X_train, y_train, _ = dp.resample('random', X_train, y_train, quake_cat_array_train, resample_size=24000 / cv.get_n_splits())

            self.best_estimator_.fit(X_train, y_train, lgb__verbose=0, lgb__eval_set=[(X_train, y_train)])
            y_sub_fold_pred = self.best_estimator_.predict(X_sub)
            y_sub_pred += y_sub_fold_pred / cv.get_n_splits()

        elif use_oof_estimate == 'prout':
          y_val_all = []
          oof_predictions = []
          for train_indexes, val_indexes in cv.split(X, quake_cat_array):
            X_train, X_val = X.iloc[train_indexes], X.iloc[val_indexes]
            y_train, y_val  = y[train_indexes], y[val_indexes]
            quake_cat_array_train = np.array(quake_cat_array)[train_indexes]

            X_train, y_train, _ = dp.resample('random', X_train, y_train, quake_cat_array_train, resample_size=2400 / cv.get_n_splits())

            self.best_estimator_.fit(X_train, y_train)
            y_val_all += list(y_val)
            y_val_pred = self.best_estimator_.predict(X_val)
            oof_predictions += list(y_val_pred)

          print('oof ', mean_absolute_error(y_val_all, oof_predictions))

        elif use_oof_estimate == 'truc':
          y_val_all = []
          oof_predictions = []

          self.best_estimator_.fit(X, y)
          print('oof ', mean_absolute_error(y, self.best_estimator_.predict(X)))


        else:
          y_sub_pred = self.best_estimator_.predict(X_sub)

        return y_sub_pred

import data_preparation as dp
def _fit_regressor(pipeline, X, y, quake_cat_array, parameters, metric, cv, random_state):

    oof_predictions = []
    val_score_per_fold = []
    y_val_all = []
    folds_weights = []
    y_pred = np.zeros(len(X))
    for train_indexes, val_indexes in cv.split(X, quake_cat_array):
        X_train, X_val = X.iloc[train_indexes], X.iloc[val_indexes]
        y_train, y_val = y[train_indexes], y[val_indexes]
        quake_cat_array_train, quake_cat_array_val = np.array(quake_cat_array)[train_indexes], np.array(quake_cat_array)[val_indexes]

        #quake_cat_array_train = dp.create_earthquake_category(X_train, y_train)
        #X, y, quake_cat_array = dp.resample('random', X, y, quake_cat_array)

        # make some data augmentation
        X_train, y_train, _ = dp.resample('random', X_train, y_train, quake_cat_array_train, resample_size=24000 / cv.get_n_splits())
        X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

        pipeline.set_params(**parameters)
        pipeline.fit(X_train, y_train, lgb__verbose=0, lgb__eval_set=[(X_train, y_train), (X_val, y_val)])

        y_val = y[val_indexes]
        y_val_pred = pipeline.predict(X_val).tolist()

        #folds_weights.append(X_val.shape[0]/X.shape[0])
        folds_weights.append(1)
        val_score_per_fold.append(metric(np.nan_to_num(y_val_pred), y_val))

        y_val_all += list(y_val)
        oof_predictions += list(y_val_pred)

    oof_score = metric(np.nan_to_num(oof_predictions), y_val_all)
    # mean_oof_score will be used to select the best parameters
    mean_oof_score = np.average(val_score_per_fold, weights=folds_weights)
    mean_oof_score = np.mean(val_score_per_fold)
    return mean_oof_score, val_score_per_fold, pipeline, parameters, oof_predictions, y_val_all, oof_score
 

def fit_model(probleme_type, X_train, y_train, pipeline, model_name, model_dict, param_dist, score_func, random_state, niter, cv, quake_cat_array, bagging_base_estimator=None, pipeline_parameters=None, refit_score='mae', n_splits=10, njobs=6,
              fit_model=False):

    scorer = make_scorer(score_func, greater_is_better=False)
    model = model_dict[model_name]
    if model_name == 'bag':
        base_estimator = model_dict[bagging_base_estimator]
        pipeline.steps.append(['bag', model(base_estimator())])
    else:
        # adding model to predefined pipeline and set parameters
        pipeline.steps.append([model_name, model()])

    if pipeline_parameters != None:
        pipeline.set_params(pipeline_parameters)

#    clf = RandomizedSearchCV(pipeline,
#                             param_distributions=param_dist,
#                             n_iter=niter,
#                             cv=cv,
#                             scoring=scorer,
#                             refit=refit_score,
#                             return_train_score=True,
#                             n_jobs=6,
#                             verbose=5)
#


    clf = CustomRandomizedSearchCV(pipeline,
                             param_distributions=param_dist,
                             n_iter=niter,
                             cv=cv,
                             quake_cat_array=quake_cat_array,
                             metric=score_func,
                             higher_is_better=False,
                             refit=refit_score,
                             return_train_score=True,
                             n_jobs=6,
                             random_state=random_state)

    clf.fit(X_train, y_train)

    return clf


def get_parameters(X, model_name, random_state, bagging_base_estimator='lgb'):

    print(model_name)
    print(bagging_base_estimator)

    if model_name == 'rf':

        param_dist = {#'rf__n_jobs': [4],
                      'rf__random_state': [random_state],
                      'rf__n_estimators': randint(100, 500),
                      'rf__criterion': ['mae', 'mse'],
                      'rf__max_depth': randint(65, 100),
                      'rf__min_samples_split': randint(2, 7),
                      'rf__min_samples_leaf': randint(2, 7),
                      #'rf__min_weight_fraction_leaf': [0],
                      'rf__max_features': randint(1, X.shape[1]),
                      'rf__max_leaf_nodes': [None],
                      #'rf__min_impurity_decrease': [0],
                      #'rf__min_impurity_split': [1e-7],
                      'rf__bootstrap': [True]}

        parameters_to_plot = ['rf__n_estimators', 'rf__min_samples_split', 'rf__max_depth', 'rf__max_features', 'rf__criterion']

    elif model_name == 'lgb':

        param_dist = {# using aliases for the parameters will display warning messages
                      'lgb__num_leaves': randint(2, 50),  # for best fit, small to qvoid overfitting. This is the most important parameter to control the complexity of the model
                      'lgb__min_child_samples': randint(1, 50),  # for best fit
                      'lgb__max_depth': randint(2, 50),  # for best fit
                      'lgb__subsample': uniform(0.05, 0.9),  # For faster speed, Deal with overfitting
                      'lgb__subsample_freq': randint(1, 50),  # For faster speed, Deal with overfitting
                      'lgb__colsample_bytree': uniform(0.05, 0.9),
                      'lgb__learning_rate': uniform(0, 1),  # For better accuracy
                      'lgb__n_estimators': randint(500, 5000),  # Big impact on computation time
                      'lgb__reg_alpha': uniform(0, 10),  # Deal with overfitting
                      'lgb__reg_lambda': uniform(0, 10),  # Deal with overfitting
                      'lgb__min_split_gain': [0],  # Deal with overfitting
                      'lgb__drop_rate': uniform(0, 1),  # Deal with overfitting
                      'lgb__objective': ['mae'],
                      #'lgb__boosting': ['gbdt'],
                      'lgb__metric': ['mae'],
                      'lgb__verbose': [-1],
                      'lgb__silent': [True],
                      'lgb__random_state': [random_state],
                      'lgb__importance_type': ['gain'], # how to calculate the features importance
                      'lgb__early_stopping_rounds': [500, 1000, 5000]


                      # 'lgb__objective': ['multiclass'], # for classification
                      # 'lgb__num_class': [3], # for classification
                      # 'lgb__metric': ['multi_logloss'] # for classification
                      }

        parameters_to_plot = ['lgb__max_depth', 'lgb__min_child_samples', 'lgb__num_leaves', 'lgb__subsample_freq', 'lgb__subsample']

    elif model_name == 'xgb':

        param_dist = {'xgb__random_state': [random_state],
                      'xgb__booster': ['gbtree'],
                      'xgb__eta': uniform(0.01, 0.5),
                      'xgb__min_child_weight': [1],
                      'xgb__max_depth': randint(1, 15),
                      #'xgb__max_leaf_nodes': randint(, ), #If this is defined, GBM will ignore max_depth
                      'xgb__gamma': uniform(0, 3),
                      #'xgb__max_delta_step ': [0],
                      'xgb__subsample': uniform(0.1, 0.9),
                      'xgb__colsample_bytree': uniform(0.1, 0.9),
                      'xgb__colsample_bylevel': uniform(0.1, 0.9),
                      'xgb__lambda': uniform(0.1, 5),
                      'xgb__alpha': uniform(0, 5),
                      #'xgb__scale_pos_weight': [1]
                      }

        parameters_to_plot = ['xgb__eta', 'xgb__max_depth', 'xgb__gamma', 'xgb__subsample', 'xgb__colsample_bytree', 'xgb__colsample_bylevel', 'xgb__lambda', 'xgb__alpha']

    elif model_name == 'svr':

        param_dist = {'svr__nu': uniform(0.8, 0.2),
                      'svr__C': uniform(5, 100),
                      'svr__kernel': ['rbf'],  # , 'linear', 'poly', 'sigmoid'],
                      #'svr__degree': randint(1, 5), # used for the plynomial kernel only
                      'svr__gamma': ['scale'],
                      'svr__coef0': uniform(0, 5),
                      'svr__tol': [1e-3],
                      'svr__shrinking': [True, False]
                      }

        parameters_to_plot = ['svr__nu', 'svr__C', 'svr__kernel', 'svr__coef0', 'svr__shrinking']

    elif model_name == 'kr':

        param_dist = {'kr__alpha': uniform(50, 80),
                      'kr__kernel': ['rbf', 'laplacian', 'linear', 'poly', 'sigmoid'],
                      'kr__gamma': uniform(20, 60),
                      'kr__degree': randint(1, 10),
                      'kr__coef0': uniform(0, 5)
                      }

        parameters_to_plot = ['kr__alpha', 'kr__kernel', 'kr__gamma', 'kr__degree', 'kr__coef0']

    elif model_name == 'stack':

        clf_names, clf_array, pipe_array = mod.get_clf_array()

        preprocessing = [StandardScaler()]

        param_dist = None

    elif model_name == 'bag':
        if bagging_base_estimator == 'svr':

            param_dist = {'bag__random_state': [random_state],
                          'bag__n_estimators': randint(5, 20),
                          'bag__max_samples': uniform(0.5, 0.5),
                          'bag__max_features': uniform(0.5, 0.5),
                          'bag__bootstrap': [True, False],
                          'bag__bootstrap_features': [True, False],
                          'bag__base_estimator__nu': uniform(0.8, 0.2),
                          'bag__base_estimator__C': uniform(1, 10),
                          'bag__base_estimator__kernel': ['rbf'],  # , 'linear', 'poly', 'sigmoid'],
                          'bag__base_estimator__gamma': ['scale'],
                          'bag__base_estimator__coef0': uniform(0, 5),
                          'bag__base_estimator__tol': [1e-3],
                          'bag__base_estimator__shrinking': [True, False]}

            parameters_to_plot = ['bag__n_estimators', 'bag__max_samples', 'bag__max_features', 'bag__bootstrap', 'bag__bootstrap_features', 'bag__base_estimator__nu', 'bag__base_estimator__C', 'bag__base_estimator__coef0', 'bag__base_estimator__shrinking']

        elif bagging_base_estimator == 'rf':

            param_dist = {'bag__random_state': [random_state],
                          'bag__n_estimators': randint(5, 20),
                          'bag__max_samples': uniform(0.5, 0.5),
                          'bag__max_features': uniform(0.5, 0.5),
                          'bag__bootstrap': [True, False],
                          'bag__bootstrap_features': [True, False],
                          'bag__base_estimator__random_state': [random_state],
                          'bag__base_estimator__n_estimators': randint(1, 200),
                          'bag__base_estimator__criterion': ['mae', 'mse'],
                          'bag__base_estimator__max_depth': randint(3, 100),
                          'bag__base_estimator__min_samples_split': randint(2, 11),
                          'bag__base_estimator__min_samples_leaf': randint(1, 11),
                          'bag__base_estimator__max_features': uniform(0.5, 0.5),
                          'bag__base_estimator__bootstrap': [True, False]}

            parameters_to_plot = ['bag__n_estimators', 'bag__max_samples', 'bag__max_features', 'bag__bootstrap', 'bag__bootstrap_features', 'bag__base_estimator__n_estimators', 'bag__base_estimator__criterion', 'bag__base_estimator__max_depth', 'bag__base_estimator__max_features']

        elif bagging_base_estimator == 'lgb':

            param_dist = {'bag__random_state': [random_state],
                          'bag__n_estimators': randint(5, 20),
                          'bag__max_samples': uniform(0.5, 0.5),
                          'bag__max_features': uniform(0.5, 0.5),
                          'bag__bootstrap': [True, False],
                          'bag__bootstrap_features': [True, False],
                          'bag__base_estimator__num_leaves': randint(2, 100),  # for best fit, small to qvoid overfitting
                          'bag__base_estimator__min_child_samples': randint(1, 100),  # for best fit
                          'bag__base_estimator__max_depth': randint(2, 500),  # for best fit
                          'bag__base_estimator__subsample': uniform(0.05, 0.9),  # For faster speed, Deal with overfitting
                          'bag__base_estimator__subsample_freq': randint(1, 50),  # For faster speed, Deal with overfitting
                          'bag__base_estimator__colsample_bytree': uniform(0.05, 0.9),
                          'bag__base_estimator__learning_rate': uniform(0, 1),  # For better accuracy
                          'bag__base_estimator__n_estimators': randint(50, 500),  # Big impact on computation time
                          'bag__base_estimator__reg_alpha': uniform(0, 10),  # Deal with overfitting
                          'bag__base_estimator__reg_lambda': uniform(0, 10),  # Deal with overfitting
                          'bag__base_estimator__min_split_gain': [0],  # Deal with overfitting
                          'bag__base_estimator__drop_rate': uniform(0, 1),  # Deal with overfitting
                          'bag__base_estimator__objective': ['mae', 'huber', 'mse'],
                          #bag__base_estimatorb__boosting': ['gbdt'],
                          'bag__base_estimator__metric': ['mae'],
                          'bag__base_estimator__verbose': [-1],
                          'bag__base_estimator__silent': [True],
                          }

            parameters_to_plot = ['bag__n_estimators', 'bag__max_samples', 'bag__max_features', 'bag__bootstrap', 'bag__bootstrap_features', 'bag__base_estimator__n_estimators', 'bag__base_estimator__max_depth', 'bag__base_estimator__min_data_in_leaf', 'bag__base_estimator__bagging_fraction']

    print(param_dist)
    return param_dist, parameters_to_plot

# ============================

'''
def fit_model_old(probleme_type, X_train, y_train, X_test, y_test, pipeline, model_name, model_dict, param_dist, score_func, quake_cat_array, pipeline_parameters=None, refit_score='mae', niter=20, n_splits=10, njobs=6,
                  stacking_meta_learner=None, stacking_preprocessing=None, bagging_base_estimator=None, fit_model=False, random_state=1, cv_by_quake=True):

    if probleme_type == 'regression':
        scorer = make_scorer(score_func=score_func, greater_is_better=False)

        model = model_dict[model_name]

        if cv_by_quake:
            # create train and test folds from our labels:
            # cv = [(np.where(quake_cat_array != quake_cat)[0], np.where(quake_cat_array == quake_cat)[0])
            #      for quake_cat in np.unique(quake_cat_array)]

            from sklearn.model_selection import PredefinedSplit
            cv = PredefinedSplit(quake_cat_array)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if model_name == 'bag':

            base_estimator = model_dict[bagging_base_estimator]

            pipeline.steps.append(['bag', model(base_estimator())])

            # pipeline.steps.append(['bag', BaggingRegressor(NuSVR())])

            clf = RandomizedSearchCV(pipeline,
                                     param_distributions=param_dist,
                                     n_iter=niter,
                                     cv=cv,
                                     scoring=scorer,
                                     refit=refit_score,
                                     return_train_score=True,
                                     n_jobs=njobs,
                                     verbose=1)

            clf.fit(X_train, y_train)

            return clf

        elif model_name == 'stack':

            # get the previously fitted model and there names into arrays
            clf_names, clf_array, _ = get_clf_array(final=True)

            # make an array of all the combination of models to take into account for the stacking
            stacked_clf_list = zip_stacked_classifiers(clf_array, clf_names)
            best_combination = [0.00, "", ""]

            # fit a stacking model for each combination of models
            for clf in stacked_clf_list:
                print(clf[0])
                ensemble = SuperLearner(scorer=score_func, random_state=random_state, folds=n_splits, n_jobs=njobs)
                ensemble.add(clf[0])
                ensemble.add_meta(meta_learner())
                # how to make a randomsearch CV for the SuperLearner ?
                ensemble.fit(X_train, y_train, preprocessing=preprocessing)
                predictions = ensemble.predict(X_test)
                score = score_func(predictions, y_test)

                if score > best_combination[0]:
                    best_combination[0] = score
                    best_combination[1] = clf[1]
                    best_combination[2] = ensemble

                print(f"Mean absolute error: {score} {clf[1]}")

            print(f"\nBest stacking model is {best_combination[1]} with mean absolute error of: {best_combination[0]}")

            return ensemble

        else:

            pipeline.steps.append([model_name, model()])

            if pipeline_parameters != None:
                pipeline.set_params(pipeline_parameters)

            clf = RandomizedSearchCV(pipeline,
                                     param_distributions=param_dist,
                                     n_iter=niter,
                                     cv=cv,
                                     scoring=scorer,
                                     refit=refit_score,
                                     return_train_score=True,
                                     n_jobs=6,
                                     verbose=10)
            clf.fit(X_train, y_train)

            return clf
    elif probleme_type == 'classification':
        scorer = make_scorer(score_func=score_func, greater_is_better=False)

        model = model_dict[model_name]

        pipeline.steps.append([model_name, model()])

        if pipeline_parameters != None:
            pipeline.set_params(pipeline_parameters)

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        clf = RandomizedSearchCV(pipeline,
                                 param_distributions=param_dist,
                                 n_iter=niter,
                                 cv=cv,
                                 scoring=scorer,
                                 refit=refit_score,
                                 return_train_score=True,
                                 n_jobs=6,
                                 verbose=10)
        clf.fit(X_train, y_train)

        return clf
'''
'''
def zip_stacked_classifiers(*args):
    to_zip = []
    for arg in args:
        combined_items = [list(list(combinations(arg, i))[0]) for i in range(len(arg) + 1)]
        combined_items.pop(0)
        to_zip.append(combined_items)

    return zip(to_zip[0], to_zip[1])
'''

# =======================================
# homemade nested CV
# =======================================
'''
def nested_CV(model, X, y, scorer, n_splits_outer, n_splits_inner, hyper_parameters, nested, random_state):

    print("Starting nested CV")

    # Performance estimation via an unbiaised estimator ---------------------------------------
    kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    outer_scores = []
    fold = 1
    hyper_parameters_list = list(product(*hyper_parameters.values()))
    hyper_parameters_keys = list(hyper_parameters.keys())

    X = X.values
    y = y.values.ravel()

    if nested == True:

        for train_index_outer, test_index_outer in kf_outer.split(X):
            X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
            y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]

            inner_mean_scores = []

            fold += 1

            for params in tqdm(hyper_parameters_list):

                inner_scores = []
                kf_inner = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)

                for train_index_inner, test_index_inner in kf_inner.split(X_train_outer):
                    # split the training data of outer CV
                    X_train_inner, X_test_inner = X_train_outer[train_index_inner], X_train_outer[test_index_inner]
                    y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]

                    scaler = StandardScaler()
                    X_train_inner = scaler.fit_transform(X_train_inner)
                    X_test_inner = scaler.transform(X_test_inner)

                    pipeline_inner = Pipeline([('model', model())])
                    # Create a dictionary of params names and params values
                    params_dict = dict(zip(hyper_parameters_keys, params))
                    # unpack the dictionary to pass it as arguments of set_params
                    pipeline_inner.set_params(**params_dict)  # un

                    # fit extremely randomized trees regressor to training data of inner CV
                    clf_inner = pipeline_inner
                    clf_inner.fit(X_train_inner, y_train_inner)
                    y_pred_inner = clf_inner.predict(X_test_inner)
                    inner_scores.append(scorer(y_test_inner, y_pred_inner))

                # calculate mean score for inner folds
                inner_mean_scores.append(np.mean(inner_scores))

            # get maximum score index
            index, value = max(enumerate(inner_mean_scores), key=operator.itemgetter(1))
            best_parameters = hyper_parameters_list[index]

            print(f"Best parameter of fold number {fold + 1} : {best_parameters}. Score : {value}")

            scaler = StandardScaler()
            X_train_outer = scaler.fit_transform(X_train_outer)
            X_test_outer = scaler.transform(X_test_outer)

            pipeline_outer = Pipeline([('model', model())])
            params_dict = dict(zip(hyper_parameters_keys, best_parameters))
            pipeline_outer.set_params(**params_dict)

            # fit the selected model to the training set of outer CV
            # for prediction error estimation
            clf_outer = pipeline_outer
            clf_outer.fit(X_train_outer, y_train_outer)
            y_pred_outer = clf_outer.predict(X_test_outer)
            outer_scores.append(scorer(y_test_outer, y_pred_outer))

        # show the prediction error estimate produced by nested CV
        print(f"Unbiased prediction error: {np.mean(outer_scores)}")

    mean_scores = []
    # Final model selection
    num_parameters = len(hyper_parameters_list)
    total_iter = num_parameters * n_splits_inner
    iteration = 1
    print(f"Final model estimation-----------------------")
    print(f"{num_parameters} sets of parameters will be tested over {n_splits_inner} folds")
    print(f"A total of {total_iter} model will be estimated")
    for params in hyper_parameters_list:
        print(f"Iteration {iteration} / {num_parameters}")
        print(params)

        scores = []

        # normal cross-validation
        kf = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
        for train_index, test_index in kf.split(X):
            # split the training data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            pipeline = Pipeline([('model', model())])
            # Create a dictionary of params names and params values
            params_dict = dict(zip(hyper_parameters_keys, params))
            # unpack the dictionary to pass it as arguments of set_params
            pipeline.set_params(**params_dict)

            # fit extremely to training data
            clf_cv = pipeline
            clf_cv.fit(X_train, y_train)
            y_pred = clf_cv.predict(X_test)
            scores.append(scorer(y_test, y_pred))

        # calculate mean score for folds
        mean_scores.append(np.mean(scores))
        iteration += 1

    # get maximum score index
    index, value = max(enumerate(mean_scores), key=operator.itemgetter(1))
    best_parameter = hyper_parameters_list[index]

    print(f"Best final parameters : {best_parameter}. Score : {value}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # finally, fit the selected model to the whole dataset
    pipeline = Pipeline([('model', model())])
    params_dict = dict(zip(hyper_parameters_keys, params))
    pipeline.set_params(**params_dict)

    clf_final = pipeline
    clf_final.fit(X, y)

    return clf_final
'''

from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso
from sklearn.feature_selection import RFECV, f_regression
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
# from minepy import MINE
import copy

class feature_selection(object):
  def __init__(self, method, scoring, random_state, pipeline=None, cv=None, group_for_cv=None, rfe_step=None, plot=True):
    self.pipeline = pipeline
    self.method = method
    self.scoring = scoring
    self.random_state = random_state
    self.cv = cv
    self.group_for_cv = group_for_cv
    self.ref_step = rfe_step
    self.plot = plot

    self.ranks_ = {}
    self.n_features_ = None
    self.support_ = None
    self.ranking_ = None
    self.grid_scores_ = None

  def fit(self, X, y):

    self.names = X.columns

    # univariate feature selection
    if self.method == 'f_reg':
      self.fit_f_reg(X, y)

    elif self.method == 'mic':
      self.fit_mic(X, y)

    # linear models and regularisation
    elif self.method ==  'linreg':
      self.fit_linreg(X, y)

    elif self.method ==  'lasso':
      self.fit_lasso(X, y)

    # models feature importance
    elif self.method ==  'rf':
      self.fit_rf(X, y)

    elif self.method ==  'lgb':
      self.fit_lgb(X, y)

    elif self.method ==  'xgb':
      self.fit_xgb(X, y)

    # stability selection and rfe
    elif self.method ==  'stability':
      self.fit_stability(X, y)

    elif self.method == 'rfe':
      self.fit_rfe(X, y, 'lgb')
      #self.fit_rfe(X, y, 'xgb')

    elif self.method ==  'all':
      self.fit_f_reg(X, y)
      #self.fit_mic(X, y)
      self.fit_linreg(X, y)
      #self.fit_lasso(X, y)
      self.fit_lgb(X, y)
      self.fit_xgb(X, y)
      self.fit_rf(X, y)
      self.fit_stability(X, y)
      self.fit_rfe(X, y)


    # Calculates the mean score of each feature through all methods
    r = {}
    for name in self.names:
      r[name] = round(np.mean([self.ranks_[method][name] 
                              for method in self.ranks_.keys()]), 5)

    methods = sorted(self.ranks_.keys())
    self.ranks_["mean"] = r
    methods.append("mean")

    print("\t%s" % "\t".join(methods))
    for name in self.names:
        print("%s\t%s" % (name, "\t".join(map(str, 
                             [self.ranks_[method][name] for method in methods]))))

    return self

  def fit_rfe(self, X, y, regressor):
    cv_splitter = self.cv.split(X, self.group_for_cv)
    if regressor == 'lgb':
      clf = LGBMRegressor(colsample_bytree=0.33, 
                          drop_rate=0.89, 
                          learning_rate=0.29, 
                          max_depth=226, 
                          metric='mae', 
                          min_child_samples=76, 
                          min_split_gain=0, 
                          n_estimators=113, 
                          num_leaves=11, 
                          objective='huber', 
                          reg_alpha=8.54, 
                          reg_lambda=0.83, 
                          silent=True, 
                          subsample=0.60, 
                          subsample_freq=36, 
                          verbose=-1,
                          random_state=self.random_state)
    elif regressor == 'xgb':
      clf = XGBRegressor(random_state=self.random_state,
               booster='gbtree',
               eta=0.5,
               min_child_weight=1,
               max_depth=15,
               gamma=2,
               max_delta_step=0,
               subsample=0.5,
               colsample_bytree=0.5,
               colsample_bylevel=0.5,
               alpha=1,
               scale_pos_weight=1)

    selector = homemade_RFECV(clf, step=self.ref_step, cv=cv_splitter, verbose=10, scoring=self.scoring, n_jobs=6)
    temp_pipeline = copy.deepcopy(self.pipeline)
    temp_pipeline.steps.append([regressor, selector])
    temp_pipeline.fit(X, y)

    self.n_features_ = selector.n_features_
    self.support_ = selector.support_
    self.ranking_ = selector.ranking_
    self.grid_scores_ = selector.grid_scores_
    self.scores_ = list(np.round(selector.scores_, 4))
    self.support_list_ = list(np.round(selector.support_list_, 4))

    if self.plot:
      # Plot number of features VS. cross-validation scores
      plt.figure()
      plt.xlabel("Number of features selected")
      plt.ylabel("Cross validation score (nb of correct classifications)")
      plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
      plt.show()




    self.ranks_["RFE"] = rank_to_dict(list(map(float, selector.ranking_)), self.names, order=-1)

  def fit_lasso(self, X, y):

    print("Extracting best features based on lasso coefficients...")

    best_alpha = None
    best_score = 1e9
    temp_pipeline = copy.deepcopy(self.pipeline)

    for alpha in np.logspace(-1.5, 20, 50):
      score_per_fold = []

      lasso = Lasso(alpha, random_state=self.random_state)
      temp_pipeline.steps.append(['lasso', lasso])

      for train_indexes, val_indexes in self.cv.split(X, self.group_for_cv):
        X_train, X_val = X.loc[train_indexes], X.loc[val_indexes]
        y_train, y_val = y[train_indexes], y[val_indexes]

        print(np.any(np.isinf(X.values)))
        print(np.any(np.isinf(y)))

        temp_pipeline.fit(X_train, y_train)
        score_per_fold.append(mean_absolute_error(temp_pipeline.predict(X_val), y_val))

      mean_oof_score = np.mean(score_per_fold)
      if mean_oof_score < best_score:
        best_score = mean_oof_score
        best_alpha = alpha

      temp_pipeline.steps.pop(1)

    print('Best parameter for lasso: ', best_alpha)
    print('Best score for lasso: ', best_score)
    lasso = Lasso(best_alpha, self.random_state)
    temp_pipeline.steps.append(['lasso', lasso])
    temp_pipeline.fit(X, y)

    self.ranks_["lasso"] = rank_to_dict(np.abs(lasso.coef_), self.names, order=-1)


  def fit_stability(self, X, y):

    print("Extracting best features based on stability scores...")

    temp_pipeline = copy.deepcopy(self.pipeline)
    rlasso = RandomizedLasso(alpha='aic',scaling=0.1, sample_fraction=0.5,  n_resampling=1000, random_state=self.random_state)
    temp_pipeline.steps.append(['rlasso', rlasso])
    temp_pipeline.fit(X, y)

    self.ranks_["stability"] = rank_to_dict(np.abs(rlasso.scores_), self.names, order=-1)


  def fit_linreg(self, X, y):

    print("Extracting best features based on linear regression coefficients...")

    temp_pipeline = copy.deepcopy(self.pipeline)
    linreg = LinearRegression()
    temp_pipeline.steps.append(['linreg', linreg])
    temp_pipeline.fit(X, y)

    self.ranks_["linreg"] = rank_to_dict(np.abs(linreg.coef_), self.names, order=-1)


  def fit_rf(self, X, y):

    print("Extracting best features based on the features importance of a random forest...")

    temp_pipeline = copy.deepcopy(self.pipeline)
    rf = RandomForestRegressor(n_jobs=6,
                      n_estimators = 200,
                      criterion = 'mae',
                      max_depth = 80,
                      min_samples_split = 5,
                      min_samples_leaf = 5,
                      min_weight_fraction_leaf=0,
                      #max_features = 100,
                      bootstrap=True,
                      oob_score=True,
                      random_state=self.random_state)
    temp_pipeline.steps.append(['rf', rf])
    temp_pipeline.fit(X,y)

    print('Best score for rf: ', rf.oob_score_)
    self.ranks_["rf"] = rank_to_dict(rf.feature_importances_, self.names, order=-1)

  def fit_lgb(self, X, y):

    print("Extracting best features based on the features importance of a lgb...")

    temp_pipeline = copy.deepcopy(self.pipeline)
    lgb = LGBMRegressor(colsample_bytree=0.33, 
                        drop_rate=0.89, 
                        learning_rate=0.29, 
                        max_depth=226, 
                        metric='mae', 
                        min_child_samples=76, 
                        min_split_gain=0, 
                        n_estimators=113, 
                        num_leaves=11, 
                        objective='huber', 
                        reg_alpha=8.54, 
                        reg_lambda=0.83, 
                        silent=True, 
                        subsample=0.60, 
                        subsample_freq=36, 
                        verbose=-1,
                        random_state=self.random_state)
    temp_pipeline.steps.append(['lgb', lgb])
    temp_pipeline.fit(X,y)

    #print('Best score for lgb: ', lgb.oob_score_)
    self.ranks_["lgb"] = rank_to_dict(lgb.feature_importances_, self.names, order=-1)


  def fit_xgb(self, X, y):

    print("Extracting best features based on the features importance of a xgb...")

    temp_pipeline = copy.deepcopy(self.pipeline)
    xgb = XGBRegressor(random_state=self.random_state,
                 booster='gbtree',
                 eta=0.5,
                 min_child_weight=1,
                 max_depth=15,
                 gamma=2,
                 max_delta_step=0,
                 subsample=0.5,
                 colsample_bytree=0.5,
                 colsample_bylevel=0.5,
                 alpha=1,
                 scale_pos_weight=1)
    temp_pipeline.steps.append(['xgb', xgb])
    temp_pipeline.fit(X,y)

    #print('Best score for xgb: ', xgb.oob_score_)
    self.ranks_["xgb"] = rank_to_dict(xgb.feature_importances_, self.names, order=-1)


 
  def fit_f_reg(self, X, y):

    print("Extracting best features based on the f score...")

    f, pval  = f_regression(X, y, center=True)
    self.ranks_["Corr."] = rank_to_dict(f, self.names)


  def fit_mic(self, X, y):

    print("Extracting best features based on mic scores...")

    mine = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        mine.compute_score(X[:,i], y)
        m = mine.mic()
        mic_scores.append(m)
 
    self.ranks_["MIC"] = rank_to_dict(mic_scores, self.names) 


def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    #ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = (order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils import check_X_y, safe_sqr
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _score
from sklearn.metrics.scorer import check_scoring
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs


def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    rfe._fit(
      X_train, y_train, lambda estimator, features:
      _score(estimator, X_test[:, features], y_test, scorer))
    return [rfe.scores_, rfe.support_list_]



class homemade_RFE(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
   
    def __init__(self, estimator, n_features_to_select=None, step=1,
                 verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, X, y, step_score=None):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit

        X, y = check_X_y(X, y, "csc")

        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        self.support_list_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            else:
                coefs = getattr(estimator, 'feature_importances_', None)
            if coefs is None:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            self.support_list_.append(list(support_))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.support_list_.append(list(support_))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_





class homemade_RFECV(homemade_RFE, MetaEstimatorMixin):

    def __init__(self, estimator, step=1, min_features_to_select=1, cv='warn',
                 scoring=None, verbose=0, n_jobs=None):
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select

    def fit(self, X, y, groups=None):

        X, y = check_X_y(X, y, "csc")

        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        rfe = homemade_RFE(estimator=self.estimator,
                  n_features_to_select=self.min_features_to_select,
                  step=self.step, verbose=self.verbose)

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        results = parallel(
            func(rfe, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y, groups))

        scores = [i[0] for i in results]
        support_list = [i[1] for i in results]
        self.scores_ = np.mean(scores, axis=0)
        self.support_list_ =  np.mean(support_list, axis=0)

        scores = np.sum(scores, axis=0) # sum scores over folds
        scores_rev = scores[::-1]
        argmax_idx = len(scores) - np.argmax(scores_rev) - 1
        n_features_to_select = max(
            n_features - (argmax_idx * step),
            self.min_features_to_select)

        # Re-execute an elimination with best_k over the whole set
        rfe = homemade_RFE(estimator=self.estimator,
                  n_features_to_select=n_features_to_select, step=self.step,
                  verbose=self.verbose)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to get_n_splits(X, y) - 1
        # here, the scores are normalized by get_n_splits(X, y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y, groups)
        return self