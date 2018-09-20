import numpy as np
from skopt import gp_minimize, gbrt_minimize
from functools import partial
import tempfile
from sklearn.externals.joblib import Parallel, delayed

# LIME
import lime
from lime import lime_tabular

# SHAP
import shap

# DEEP Explain
import keras.models

# L2X

from L3Xutils import L3X


def make_keras_picklable():
    def __getstate__(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)

        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def _parallel_lipschitz(wrapper, i, x, bound_type, eps, n_calls):
    make_keras_picklable()
    print('\n\n ***** PARALLEL : Example ' + str(i) + '********')
    print(wrapper.net.__dict__.keys())
    if 'model' in wrapper.net.__dict__.keys():
        lip_fcn, _ = wrapper.local_lipschitz_estimate(
            x, eps=eps, bound_type=bound_type, n_calls=n_calls)
    else:
        lip_fcn = None
    return lip_fcn


class ExplainerWrapper(object):
    def __init__(self, model, mode, explainer, multiclass=False,
                 feature_names=None, class_names=None, train_data=None):
        self.mode = mode
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
        self.multiclass = multiclass
        self.class_names = class_names
        self.train_data = train_data  # Necessary only to get data distrib stats

        if self.train_data is not None:
            # These are per-feature dim statistics
            # print("Computing train data stats...")
            self.train_stats = {
                'min': self.train_data.min(0),
                'max': self.train_data.max(0),
                'mean': self.train_data.mean(0),
                'std': self.train_data.std(0)
            }
            # pprint(self.train_stats)

    def estimate_dataset_lipschitz(self, dataset, continuous=True, eps=1, maxpoints=None,
                                   optim='gp', bound_type='box', n_jobs=1, n_calls=10,
                                   verbose=False):
        """
            Continuous and discrete space version.
        """
        make_keras_picklable()
        dataset_length = len(dataset)
        if maxpoints and dataset_length > maxpoints:
            dataset_filt = dataset[np.random.choice(dataset_length, maxpoints)]
        else:
            dataset_filt = dataset[:]
        if n_jobs > 1:
            lips = Parallel(n_jobs=n_jobs, max_nbytes=1e6, verbose=verbose)(
                delayed(_parallel_lipschitz)(self, i=i, x=x, bound_type=bound_type, eps=eps,
                                             n_calls=n_calls) for i, x in enumerate(dataset_filt))
        else:
            lips = []
            for datapoint in dataset_filt:
                l, _ = self.local_lipschitz_estimate(
                    datapoint, optim=optim, bound_type=bound_type, eps=eps, n_calls=n_calls,
                    verbose=verbose)
                lips.append(l)
        print(
            'Missed points: {}/{}'.format(sum(x is None for x in lips), len(dataset_filt)))
        lips = np.array([l for l in lips if l is not None])
        return lips

    def lipschitz_ratio(self, x0=None, x1=None, minus=False):
        """
            If minus = True, returns minus this quantitiy.
            || f(x0) - f(x1) ||/||x0 - x1||
        """
        # Need this because skopt sends lists
        if type(x0) is list:
            x0 = np.array(x0)
        if type(x1) is list:
            x1 = np.array(x1)
        if x0.shape[0] > 1:
            x0 = x0.reshape(1, -1)
        if x1.shape[0] > 1:
            x1 = x1.reshape(1, -1)
        multip = -1 if minus else 1
        return multip * np.linalg.norm(self(x0) - self(x1)) / np.linalg.norm(x0 - x1)

    def local_lipschitz_estimate(self, x, optim='gp', eps=None, bound_type='box',
                                 clip=True, n_calls=100, verbose=False):
        """
            Compute one-sided lipschitz estimate for explainer. Adequate for local Lipschitz,
            for global must have the two-sided version. This computes:
                max_z || f(x) - f(z)|| / || x - z||
            Instead of:
                max_z1,z2 || f(z1) - f(z2)|| / || z1 - z2||
            If eps provided, does local lipzshitz in:
                - box of width 2*eps along each dimension if bound_type = 'box'
                - box of width 2*eps*va, along each dimension if bound_type = 'box_norm'
                (i.e. normalize so that deviation is eps % in each dim )
                - box of width 2*eps*std along each dimension if bound_type = 'box_std'
            max_z || f(x) - f(z)|| / || x - z||   , with f = theta
            clip: clip bounds to within (min, max) of dataset
        """
        assert optim in ['gbrt', 'gp'], "Presently supported optimisation types are 'gbrt' or 'gp'."
        # Compute bounds for optimization
        if eps is None:
            # For global lipzhitz ratio maximizer  - use max min bounds of dataset fold of interest
            # gp can't have lower bound equal upper bound - so move them slightly appart
            lwr = self.train_stats['min'].flatten() - 1e-6
            upr = self.train_stats['max'].flatten() + 1e-6
        elif bound_type == 'box':
            lwr = (x - eps).flatten()
            upr = (x + eps).flatten()
        elif bound_type == 'box_std':
            # gp can't have lower bound equal upper bound - so set min std to 0.001
            lwr = (x - eps * np.maximum(self.train_stats['std'], 0.001)).flatten()
            upr = (x + eps * np.maximum(self.train_stats['std'], 0.001)).flatten()
        if clip:
            lwr = lwr.clip(min=self.train_stats['min'].min())
            upr = upr.clip(max=self.train_stats['max'].max())
        bounds = list(zip(*[lwr, upr]))

        # Run optimization
        if optim == 'gbrt':
            print('Running BlackBox Minimization with Gradient Boosted Trees')
            func = partial(self.lipschitz_ratio, x, minus=True)
            res = gbrt_minimize(func, bounds, n_calls=n_calls, verbose=verbose, n_jobs=-1)
        else:
            print('Running BlackBox Minimization with Bayesian Optimization')
            # Need minus because gp only has minimize() method
            func = partial(self.lipschitz_ratio, x, minus=True)
            res = gp_minimize(func, bounds, n_calls=n_calls, verbose=verbose, n_jobs=-1)

        lip, x_opt = -res['fun'], np.array(res['x'])
        if verbose:
            print(lip, np.linalg.norm(x - x_opt))
        return lip, x_opt

        # def estimate_discrete_dataset_lipschitz(self, dataset, eps=None, top_k=1,
        # metric='euclidean'):
        #     """
        #         For every point in dataset, find pair point y in dataset that maximizes
        #         Lipschitz: || f(x) - f(y) ||/||x - y||
        #         Args:
        #             - dataset: a tds obkect
        #             - top_k : how many to return
        #             - max_distance: maximum distance between points to consider (radius)
        #     """
        #     Xs  = dataset
        #     n, d = Xs.shape
        #     Fs = self(Xs)
        #     num_dists = pairwise_distances(Fs)  # metric = 'euclidean')
        #     den_dists = pairwise_distances(Xs, metric=metric)  # Or chebyshev?
        #     if eps is not None:
        #         nonzero = np.sum((den_dists > eps))
        #         total   = den_dists.size
        #         print('Number of zero denom distances: {} ({:4.2f}%)'.format(
        #             total - nonzero, 100*(total-nonzero)/total))
        #         den_dists[den_dists > eps] = -1.0   # float('inf')
        #     # Same with self dists
        #     den_dists[den_dists == 0] = -1  # float('inf')
        #     ratios = (num_dists/den_dists)
        #     argmaxes = {k: [] for k in range(n)}
        #     vals, inds = topk_argmax(ratios, top_k)
        #     argmaxes = {i:  [(j, v) for (j, v) in zip(inds[i, :], vals[i, :])] for i in range(n)}
        #     return vals.squeeze(), argmaxes


class ShapWrapper(ExplainerWrapper):
    """
        Wrapper around SHAP explanation framework from shap github package by the authors
    """

    def __init__(self, model, shap_type, link, mode, multiclass=False, feature_names=None,
                 class_names=None, train_data=None, num_features=None, categorical_features=None,
                 nsamples=100, verbose=False):
        assert shap_type in ['kernel', 'tree'], \
            "Presently supported SHAP explainer types are 'kernel' and 'tree'."
        print('Initializing {} SHAP explainer wrapper'.format(shap_type))
        super().__init__(model, mode, None, multiclass,
                         feature_names, class_names, train_data)
        if shap_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model, train_data, link=link)

        self.explainer = explainer
        self.nsamples = nsamples

    def __call__(self, x, y=None, x_raw=None, return_dict=False, show_plot=False):
        """
            y only needs to be specified in the multiclass case. In that case,
            it's the class to be explained (typically, one would take y to be
            either the predicted class or the true class). If it's a single value,
            same class explained for all inputs
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert hasattr(self.model, 'predict_proba') | hasattr(self.model, 'predict'), \
            'XAI currently only supports models with predict or predict_proba attributes'

        if self.mode == 'regression':
            try:
                predict_fn = self.model.predict
                y = np.argmax(predict_fn(x))
            except AttributeError:
                raise AttributeError("Regression model object supplied has no predict() method")
            exp = self.explainer.shap_values(x, nsamples=self.nsamples, verbose=False)
            exp_dict = dict(zip(self.feature_names + ['bias'], exp.T.tolist()))
        elif self.mode == 'classification':
            try:
                predict_fn = self.model.predict_proba
                y = np.argmax(predict_fn(x).reshape(x.shape[0], len(self.class_names)), axis=1)
            except AttributeError:
                raise AttributeError(
                    "Classification model object supplied has no predict_proba() method")
            assert x.shape[0] == len(y), "x.shape = {}, len(y) = {}".format(x.shape, len(y))
            exp = self.explainer.shap_values(x, nsamples=self.nsamples, verbose=False)
            exp_dict = dict(zip(self.feature_names + ['bias'], exp[0].T.tolist()))

        # if x.shape[0] == 1:
        #     # Squeeze if single prediction
        #     exp = exp[0]

        self.explanation = exp
        vals = np.array([exp_dict[feat]
                         for feat in self.feature_names if feat in exp_dict.keys()]).T
        if not return_dict:
            return vals
        else:
            return exp_dict


class LimeWrapper(ExplainerWrapper):
    """
        Wrapper around LIME explanation framework from lime github package by the authors
    """

    def __init__(self, model, lime_type, mode, multiclass=False, feature_names=None,
                 num_samples=100, class_names=None, train_data=None, num_features=None,
                 categorical_features=None, verbose=False):
        assert lime_type == 'tabular', "Currently supported LIME type is 'tabular'."
        print('Initializing {} LIME explainer wrapper'.format(lime_type))

        explainer = lime.lime_tabular.LimeTabularExplainer(
            train_data, feature_names=feature_names, class_names=class_names,
            discretize_continuous=False, categorical_features=categorical_features,
            verbose=verbose, mode=mode)

        super().__init__(model, mode, explainer, multiclass,
                         feature_names, class_names, train_data)
        self.lime_type = lime_type
        # self.explainer = explainer
        self.num_features = num_features if num_features else len(
            self.feature_names)
        self.num_samples = num_samples

    def extract_att_tabular(self, exp, y):
        # """ Method for extracting a numpy array from lime explanation objects"""
        if self.mode == 'classification':
            if y is None or (type(y) is list and y[0] is None):
                exp_dict = dict(exp.as_list(exp.top_labels[0]))
            elif self.multiclass:
                exp_dict = dict(exp.as_list(label=y))
            else:
                exp_dict = dict(exp.as_list(exp.top_labels[0]))

        if self.mode == 'regression':
            exp_dict = dict(exp.as_list())

        vals = np.array([exp_dict[feat] for feat in self.feature_names if feat in exp_dict.keys()])
        return vals

    def __call__(self, x, y=None, x_raw=None, return_dict=False, show_plot=False):
        # if y is None:
        #     # No target class provided - use predicted
        #     y = self.model(x).argmax(1)

        assert self.lime_type == 'tabular', "XAI currently supports only Tabular LIME"
        if (self.mode == 'classification') & (not hasattr(self.model, 'predict_proba')):
            raise AttributeError("LIME does not currently support classifier models without "
                                 "probability scores. If this conflicts with your use case, refer "
                                 "to this issue: https://github.com/datascienceinc/lime/issues/16")
        if x.ndim == 1:
            if (self.mode == 'classification') & hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
                if not self.multiclass:
                    labs, top_labs = [(1,)], None  # Explain the "only" class
                elif y is None:
                    labs, top_labs = None, 1  # Explain only most likely predicted class
                else:
                    labs, top_labs = (y,), None  # Explains class y

            elif (self.mode == 'regression') & hasattr(self.model, 'predict'):
                predict_fn = self.model.predict
                top_labs = None
                labs = None

            exp = self.explainer.explain_instance(x, predict_fn=predict_fn,
                                                  labels=labs,
                                                  top_labels=top_labs,
                                                  num_features=self.num_features,
                                                  num_samples=self.num_samples,
                                                  distance_metric='euclidean')
            self.explanation = exp
            attributions = self.extract_att_tabular(exp, y)
        else:
            # There's multiple examples to explain
            n = int(x.shape[0])
            if (self.mode == 'classification') & hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
                if not self.multiclass:
                    labs = [(1,)] * n
                    top_labs = 1
                elif y is None:
                    labs = [(None,)] * n
                    top_labs = 1
                else:
                    top_labs = None
                    labs = [(y[i],) for i in range(n)]

            elif (self.mode == 'regression') & hasattr(self.model, 'predict'):
                predict_fn = self.model.predict
                top_labs = None
                labs = [(None,)] * n

            exp = [self.explainer.explain_instance(x[i, :], predict_fn=predict_fn,
                                                   labels=labs[i], top_labels=top_labs,
                                                   num_features=self.num_features,
                                                   num_samples=self.num_samples) for i in range(n)]
            self.explanation = exp
            attributions = [self.extract_att_tabular(exp[i], labs[i][0]) for i in range(len(exp))]
            attributions = np.stack(attributions, axis=0)

        return attributions

class L2XWrapper(ExplainerWrapper):
    """
    Wrapper for provided L2X model; assuming L2X model's predictions are np arrays of the form batch x dim


    """
    def __init__(self, model, explainer=None, L2X_type = 'standard', k=2, epochs=10, tau=0.5, train=None, 
                val=None, batch_size=10, num_features=None, verbose=False, multiclass=None, feature_names=None, 
                class_names=None, train_data=None):

        """
        INPUTS:
        model -- a black box model that is already trained. Should have a 'predict' method that takes n_instances x dim 
                    and returns n_instances x num_classes (standard L2X is trained to explain classifiers)
        explainer -- either a pre-trained L2X model or None. If None, wrapper will attempt to train a new L2X model. If
                    L2X is being trained afresh, train_data and val_data should not be none
        L2X_type -- string. 'standard' is all that is currently supported
        k -- integer. Number of features for L2X to train to select.
        epochs -- integer. Training epochs for L2X
        tau -- float. Governs degree of relaxation of hard sampling.
        train -- array. Should have dimensions n_instances x feature dimension
        val -- array. Should have dimensions n_instances x feature dimension
        batch_size -- integer. Choose appropriate batch size during training. Note current BUG: if train and val % batch size
                    =/= 0, L2X will throw a broadcasting error.
        num_features -- integer.
        verbose -- Bool.

        ASSUMING train_data does something different to what I want train and val to do.

        """

        assert L2X_type == 'tabular', "Currently supported L2X type is 'standard'."
        assert hasattr(model, 'predict'), 'L2X doesnt support models that dont support a predict method'

        print('Initializing {} L2X explainer wrapper'.format(L2X_type))

        if not explainer: #This will be sloooooow unless you're doing it on a GPU
            pred_train = model.predict(train)
            pred_val = model.predict(val)

            __, explainer = L3X([train_data, pred_train, val_data, pred_val], batch_size)

        super().__init__(model, mode, explainer, multiclass,
                         feature_names, class_names, train_data)
        self.num_features = train.shape[1]
        self.batch_size = batch_size

    def __call__(self, x, y=None, x_raw=None, return_dict=False, show_plot=False):
        if x.ndim == 1:
            attributions = explainer.predict(np.expand_dims(x, 0), verbose=int(verbose), batch_size=batch_size)

        else:
            attributions = explainer.predict(x, verbose=int(verbose), batch_size=batch_size)

        return attributions


