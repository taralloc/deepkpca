# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
# https://github.com/google-research/disentanglement_lib
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CHANGES:
# - converted from tensorflow to pytorch
# - removed gin config
# - uses disent objects and classes
# - renamed functions


import logging

import numpy as np
from tqdm import tqdm

from disent.dataset import DisentDataset
from disent.metrics import utils
from disent.metrics.utils import make_metric
from disent.util import to_numpy


log = logging.getLogger(__name__)


# ========================================================================= #
# factor_vae                                                                #
# ========================================================================= #


@make_metric('irs', fast_kwargs=dict(num_train=700,  num_eval=350))
def metric_irs(
        dataset: DisentDataset,
        representation_function: callable,
        batch_size: int = 64,
        num_train: int = 10000,
        num_eval: int = 5000,
        diff_quantile=0.99,
        available_y=None
):
    """
    Computes the IRS disentanglement metric.

    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      batch_size: Number of points to be used to compute the training_sample.
      num_train: Number of points used for training.
      num_eval: Number of points used for evaluation.
      num_variance_estimate: Number of points used to estimate global variances.
      show_progress: If a tqdm progress bar should be shown
    Returns:
      Dictionary with scores:
        train_accuracy: Accuracy on training set.
        eval_accuracy: Accuracy on evaluation set.
    """

    log.debug("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(dataset, representation_function, num_train, batch_size, available_y=available_y)
    assert mus_train.shape[1] == num_train

    ys_discrete = make_discretizer(ys_train)
    active_mus = _drop_constant_dims(mus_train)

    if not active_mus.any():
        irs_score = 0.0
    else:
        irs_score = scalable_disentanglement_score(ys_discrete.T, active_mus.T,
                                                   diff_quantile)["avg_score"]

    return {
        "irs": irs_score,    # "z-min variance" -- Measuring Disentanglement: A Review of Metrics
    }


def _prune_dims(variances, threshold=0.):
    """Mask for dimensions collapsed to the prior."""
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def _compute_variances(
        dataset: DisentDataset,
        representation_function: callable,
        batch_size: int,
        eval_batch_size: int = 64
):
    """Computes the variance for each dimension of the representation.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observation as input and outputs a representation.
      batch_size: Number of points to be used to compute the variances.
      eval_batch_size: Batch size used to eval representation.
    Returns:
      Vector with the variance of each dimension.
    """
    observations = dataset.dataset_sample_batch(batch_size, mode='input')
    representations = to_numpy(utils.obtain_representation(observations, representation_function, eval_batch_size))
    representations = np.transpose(representations)
    assert representations.shape[0] == batch_size
    return np.var(representations, axis=0, ddof=1)


def _generate_training_sample(
        dataset: DisentDataset,
        representation_function: callable,
        batch_size: int,
        global_variances: np.ndarray,
        active_dims: list,
) -> (int, int):
    """Sample a single training sample based on a mini-batch of ground-truth data.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observation as input and
        outputs a representation.
      batch_size: Number of points to be used to compute the training_sample.
      global_variances: Numpy vector with variances for all dimensions of representation.
      active_dims: Indexes of active dimensions.
    Returns:
      factor_index: Index of factor coordinate to be used.
      argmin: Index of representation coordinate with the least variance.
    """
    # Select random coordinate to keep fixed.
    factor_index = np.random.randint(dataset.gt_data.num_factors)
    # Sample two mini batches of latent variables.
    factors = dataset.gt_data.sample_factors(batch_size)
    # Fix the selected factor across mini-batch.
    factors[:, factor_index] = factors[0, factor_index]
    # Obtain the observations.
    observations = dataset.dataset_batch_from_factors(factors, mode='input')
    representations = to_numpy(representation_function(observations))
    local_variances = np.var(representations, axis=0, ddof=1)
    argmin = np.argmin(local_variances[active_dims] / global_variances[active_dims])
    return factor_index, argmin


def _generate_training_batch(
        dataset: DisentDataset,
        representation_function: callable,
        batch_size: int,
        num_points: int,
        global_variances: np.ndarray,
        active_dims: list,
        show_progress=False,
):
    """Sample a set of training samples based on a batch of ground-truth data.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      batch_size: Number of points to be used to compute the training_sample.
      num_points: Number of points to be sampled for training set.
      global_variances: Numpy vector with variances for all dimensions of representation.
      active_dims: Indexes of active dimensions.
    Returns:
      (num_factors, dim_representation)-sized numpy array with votes.
    """
    votes = np.zeros((dataset.gt_data.num_factors, global_variances.shape[0]), dtype=np.int64)
    for _ in tqdm(range(num_points), disable=(not show_progress)):
        factor_index, argmin = _generate_training_sample(dataset, representation_function, batch_size, global_variances, active_dims)
        votes[factor_index, argmin] += 1
    return votes

def _drop_constant_dims(ys):
  """Returns a view of the matrix `ys` with dropped constant rows."""
  ys = np.asarray(ys)
  if ys.ndim != 2:
    raise ValueError("Expecting a matrix.")

  variances = ys.var(axis=1)
  active_mask = variances > 0.
  return ys[active_mask, :]


def scalable_disentanglement_score(gen_factors, latents, diff_quantile=0.99):
  """Computes IRS scores of a dataset.
  Assumes no noise in X and crossed generative factors (i.e. one sample per
  combination of gen_factors). Assumes each g_i is an equally probable
  realization of g_i and all g_i are independent.
  Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).
  Returns:
    Dictionary with IRS scores.
  """
  num_gen = gen_factors.shape[1]
  num_lat = latents.shape[1]

  # Compute normalizer.
  max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
  cum_deviations = np.zeros([num_lat, num_gen])
  for i in range(num_gen):
    unique_factors = np.unique(gen_factors[:, i], axis=0)
    assert unique_factors.ndim == 1
    num_distinct_factors = unique_factors.shape[0]
    for k in range(num_distinct_factors):
      # Compute E[Z | g_i].
      match = gen_factors[:, i] == unique_factors[k]
      e_loc = np.mean(latents[match, :], axis=0)

      # Difference of each value within that group of constant g_i to its mean.
      diffs = np.abs(latents[match, :] - e_loc)
      max_diffs = np.percentile(diffs, q=diff_quantile*100, axis=0)
      cum_deviations[:, i] += max_diffs
    cum_deviations[:, i] /= num_distinct_factors
  # Normalize value of each latent dimension with its maximal deviation.
  normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
  irs_matrix = 1.0 - normalized_deviations
  disentanglement_scores = irs_matrix.max(axis=1)
  if np.sum(max_deviations) > 0.0:
    avg_score = np.average(disentanglement_scores, weights=max_deviations)
  else:
    avg_score = np.mean(disentanglement_scores)

  parents = irs_matrix.argmax(axis=1)
  score_dict = {}
  score_dict["disentanglement_scores"] = disentanglement_scores
  score_dict["avg_score"] = avg_score
  score_dict["parents"] = parents
  score_dict["IRS_matrix"] = irs_matrix
  score_dict["max_deviations"] = max_deviations
  return score_dict

def histogram_discretizer(target, num_bins=20):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized

def make_discretizer(target, num_bins=20, discretizer_fn=histogram_discretizer):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
