import tempfile

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
from sklearn.model_selection import train_test_split

# These are default values provided by the original colab
import random
# random_seed = random.randint(0,10000)
# print(f"{random_seed=}")
random_seed = 2 # @param {type: "integer"}
batch_size = 64 # @param {type: "integer"}
learning_rate = 0.001 # @param {type: "float"}
num_training_steps = 50_000 # @param {type: "integer"}
validation_interval = 100 # @param {type: "integer"}

prngkey = 1 # @param {type: "integer"}

# @title Download data
import pandas as pd
import os
#_, input_filename = tempfile.mkstemp()
input_filename = "./knot_theory_invariants.csv"
# USE YOUR OWN gsutil cp COMMAND HERE
if not os.path.exists(input_filename):
    os.system(f"~/software/google-cloud-sdk/bin/gsutil cp gs://maths_conjectures/knot_theory/knot_theory_invariants.csv {input_filename}")

full_df = pd.read_csv(input_filename)

# @title Load and preprocess data

#@markdown The columns of the dataset which will make up the inputs to the network.
#@markdown In other words, for a knot k, X(k) will be a vector consisting of these quantities. In this case, these are the geometric invariants of each knot.
#@markdown For descriptions of these invariants see https://knotinfo.math.indiana.edu/
display_name_from_short_name = {
    'chern_simons': 'Chern-Simons',
    'cusp_volume': 'Cusp volume',
    'hyperbolic_adjoint_torsion_degree': 'Adjoint Torsion Degree',
    'hyperbolic_torsion_degree': 'Torsion Degree',
    'injectivity_radius': 'Injectivity radius',
    'longitudinal_translation': 'Longitudinal translation',
    'meridinal_translation_imag': 'Re(Meridional translation)',
    'meridinal_translation_real': 'Im(Meridional translation)',
    'short_geodesic_imag_part': 'Im(Short geodesic)',
    'short_geodesic_real_part': 'Re(Short geodesic)',
    'Symmetry_0': 'Symmetry: $0$',
    'Symmetry_D3': 'Symmetry: $D_3$',
    'Symmetry_D4': 'Symmetry: $D_4$',
    'Symmetry_D6': 'Symmetry: $D_6$',
    'Symmetry_D8': 'Symmetry: $D_8$',
    'Symmetry_Z/2 + Z/2': 'Symmetry: $\\frac{Z}{2} + \\frac{Z}{2}$',
    'volume': 'Volume',
}
column_names = list(display_name_from_short_name)
target = 'signature'

#@markdown Split the data into a training, a validation and a holdout test set. To check
#@markdown the robustness of the model and any proposed relationship, the training
#@markdown process can be repeated with multiple different train/validation/test splits.

#@markdown Calculate the mean and standard deviation over each column in the training
#@markdown dataset. We use this to normalize each feature, this is best practice for
#@markdown inputting features into a network, but is also very important in this case
#@markdown to ensure the gradients used for saliency are meaningfully comparable.
def normalize_features(df, cols, add_target=True):
  features = df[cols]
  sigma = features.std()
  if any(sigma == 0):
    print(sigma)
    raise RuntimeError(
        "A poor data stratification has led to no variation in one of the data "
        "splits for at least one feature (ie std=0). Restratify and try again.")
  mu = features.mean()
  normed_df = (features - mu) / sigma
  if add_target:
    normed_df[target] = df[target]
  return normed_df


def get_batch(df, cols, size=None):
  batch_df = df if size is None else df.sample(size)
  X = batch_df[cols].to_numpy()
  y = batch_df[target].to_numpy()
  return X, y



random_state = np.random.RandomState(random_seed)
train_df, validation_and_test_df = train_test_split(
    full_df, random_state=random_state)
validation_df, test_df = train_test_split(
    validation_and_test_df, test_size=.5, random_state=random_state)


# normed_train_df = normalize_features(train_df, column_names)
# Sometimes this line will cause a RuntimeError when running smaller networks. Just restart the kernel and try again.
# normed_validation_df = normalize_features(validation_df, column_names)
normed_test_df = normalize_features(test_df, column_names)

test_X, test_y = get_batch(normed_test_df, column_names)



# Find bounds for the signature in the training dataset.
max_signature = train_df[target].max()
min_signature = train_df[target].min()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-hn", type=int, default=3)
args = parser.parse_args()
hid_dim = args.hn

def compacted_net_forward(inp):
  return hk.Sequential([
      hk.Linear(hid_dim),
      jax.nn.sigmoid,
      hk.Linear(int((max_signature - min_signature) / 2)+1),
  ])(inp)
compacted_net_forward_t = hk.without_apply_rng(hk.transform(compacted_net_forward))

@jax.jit
def compacted_predict(params, data_X):
  return (np.argmax(compacted_net_forward_t.apply(params, data_X), axis=1) * 2 +
          min_signature)

if hid_dim == 2:
    num_ckpt = 78
elif hid_dim == 3:
    num_ckpt = 110

import pickle
with open(f'ckpts/{num_ckpt}.pickle', 'rb') as f:
    hk_params = pickle.load(f)
    print(np.mean((compacted_predict(hk_params, test_X) - test_y) == 0))