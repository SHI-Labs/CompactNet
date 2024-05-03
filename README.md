# CompactNet
This repo is a slight modification on DeepMind's _Advancing mathematics by
guiding human intuition with AI_
- [Original Notebook](https://colab.research.google.com/github/deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb)
- [Original Repo](https://github.com/google-deepmind/mathematics_conjectures)
- [Nature Paper](https://www.nature.com/articles/s41586-021-04086-x)

We noticed some issues when the [KAN](https://arxiv.org/abs/2404.19756) paper 
cited this work and found that the comparisons had some errors.
We found that we could match KAN's 81.6% accuracy on this dataset with as few as
122 parameters.
We did not make any major modifications to the DeepMind code.
To achieve this result we only decreased the network size, used a random seed,
and increased the training time.
Keeping the same seed and keeping the same training cutoff we could get a
matching result with a network with 204 parameters.

The table below depicts some of our results.
There are some variances so numbers may change slightly during your runs.
Running several times you should be quite similar to ours.
These results maintain the same random seed and the same training limit.

| Network | Number of Neurons| Number of Parameters | Accuracy Pre Salient | Accuracy Post Salient |
|:--------:|:---------------------:|:---------------------:|:----------------------:|:-----------------------:|
| [300, 300, 300] | 900 | 190,214 | 81.38% | 80.14% |
| [100, 100, 100] | 300 | 23,414 | 82.79% | 82.04% |
| [50, 50, 50] | 150 | 6,714 | 85.13% | 81.65% |
| [10, 10, 10] | 30 | 554 | 84.45% | 82.30% |
| [5, 5, 5] | 15 | 234 | 83.06% | 80.42% |
| [4, 4, 4] | 12 | 182 | 76.73% | 65.19% |
| [3, 3, 3] | 9 | 134 | 66.33% | 74.93% |
| [50, 50] | 100 | 4,164 | 87.15% | 82.65% |
| [10, 10] | 20 | 444 | 83.02% | 81.50% |
| [5, 5] | 10 | 204 | 82.19% | 81.33% |
| [4, 4] | 8 | 162 | 81.89% | 81.03% |
| [3, 3] | 6 | 122 | 77.72% | 76.24% |
| Baseline (direct calculate) | 0 | 0 | 73.82% | 73.82% |
| DeepMind's 4 layer reported | 900 | 190,214 | 78% | 78% |
| KAN | 32 | 200 | 81.6% | 78.2% |

Here are some results where we have changed the random seed and training length. 
We set `num_training_steps` to 50k for an arbitrarially long run and report how many steps before the network early stopped (`Steps`)
| Network | Number of Neurons|Number of Parameters | seed | Accuarcy Pre Salient | Steps | Accuracy Post Salient | Steps |
|:--------:|:----:|:---------------------:|:----:|:---------------------:|:-----:|:----------------------:|:-----:|
| [3,3] | 6 | 122 | 552 | 81.60% | 20700 | 81.69% | 22100 |
| [2,2] | 4 | 84 | 8110 | 81.33%  | 22700 | 80.44% | 23300 |

We also have advanced methods to train a two layer extremely small MLP in 1k steps that can achieve average performance (averaged over 10 times) of more thann 80%. You may check the models in the folder `ckpt` for details. You may run the following code for inference.
```python
python test_ckpt.py -hn 3
python test_ckpt.py -hn 2
```

| Network | Number of Neurons|Number of Parameters | Accuarcy Pre Salient | Steps | Accuracy Post Salient | Steps |
|:--------:|:---------------------:|:----:|:---------------------:|:-----:|:----------------------:|:-----:|
| [3] | 3 | 110 | 80.85% | ~1000 | 80.43% | ~1000 |
| [2] | 2 | 78  | 82.42% | ~1000 | 80.33% | ~1000 |


## Running
```
pip install -r requirements.txt
```
You will also need to install the dataset which requires having installed 
[gsutil](https://cloud.google.com/storage/docs/gsutil_install).
If you install this, we will automatically download the dataset for you.
Make sure `gsutil` is in your path before opening the notebook or it may not be
able to download it for you.

Line 1 of the file contains the network definition where you should define how
many hidden neurons you want per hidden layer.
The length of the list determines the number of hidden layers.
For example `[2,2]` means two hidden layers with 2 neurons each.
