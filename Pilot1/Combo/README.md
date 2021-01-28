## Combo: Predicting Tumor Cell Line Response to Drug Pairs

**Overview**: Given combination drug screening results on NCI60 cell lines available at the NCI-ALMANAC database, build a deep learning network that can predict the growth percentage from the cell line molecular features and the descriptors of both drugs.

**Relationship to core problem**: This benchmark is an example one of the core capabilities needed for the Pilot 1 Drug Response problem: combining multiple molecular assays and drug descriptors in a single deep learning framework for response prediction.

**Expected outcome**: Build a DNN that can predict growth percentage of a cell line treated with a pair of drugs.

### Benchmark Specs Requirements

#### Description of the Data
* Data source: Combo drug response screening results from NCI-ALMANAC; 5-platform normalized expression, microRNA expression, and proteome abundance data from the NCI; Dragon7 generated drug descriptors based on 2D chemical structures from NCI
* Input dimensions: ~30K with default options: 26K normalized expression levels by gene + 4K drug descriptors; 59 cell lines; a subset of 54 FDA-approved drugs
Output dimensions: 1 (growth percentage)
* Sample size: 85,303 (cell line, drug 1, drug 2) tuples from the original 304,549 in the NCI-ALMANAC database
* Notes on data balance: there are more ineffective drug pairs than effective pairs; data imbalance is somewhat reduced by using only the best dose combination for each (cell line, drug 1, drug 2) tuple as training and validation data

#### Expected Outcomes
* Regression. Predict percent growth for any NCI-60 cell line and drugs combination
* Dimension: 1 scalar value corresponding to the percent growth for a given drug concentration. Output range: [-100, 100]

#### Evaluation Metrics
* Accuracy or loss function: mean squared error, mean absolute error, and R^2
* Expected performance of a naïve method: mean response, linear regression or random forest regression.

#### Description of the Network
* Proposed network architecture: two-stage neural network that is jointly trained for feature encoding and response prediction; shared submodel for each drug in the pair
* Number of layers: 3-4 layers for feature encoding submodels and response prediction submodels, respectively

### Setup:
To setup the python environment needed to train and run this model, first make sure you install [conda](https://docs.conda.io/en/latest/) package manager, clone this repository, then create the environment as shown below.

```bash
   conda env create -f environment.yml -n Combo
   conda activate Combo
   ```

### Running the baseline implementation

```
$ cd Pilot1/Combo
$ python combo_baseline_keras2.py --cell_features rnaseq --drug_features descriptors --residual True --cp True --epochs 100 --use_landmark_genes True --warmup_lr True --reduce_lr True --base_lr 0.0003 -z 128 --preprocess_rna source_scale
```

#### Example output
```
Comparing y_true and y_pred:
  mse: 0.0382
  mae: 0.1339
  r2: 0.8557
  corr: 0.9254
```

#### Inference

There is a separate inference script that can be used to predict drug pair response on combinations of sample sets and drug sets with a trained model.
```
$ python infer.py --sample_set NCIPDM --drug_set ALMANAC

Using TensorFlow backend.
Predicting drug response for 6381440 combinations: 590 samples x 104 drugs x 104 drugs
100%|██████████████████████████████████████████████████████████████████████| 639/639 [14:56<00:00,  1.40s/it]
```
Example trained model files can be downloaded here: [saved.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.model.h5) and [saved.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.weights.h5).

The inference script also accepts models trained with [dropout as a Bayesian Approximation](https://arxiv.org/pdf/1506.02142.pdf) for uncertainty quantification. Here is an example command line to make 100 point predictions for each sample-drugs combination in a subsample of the GDSC data.

```
$ python infer.py -s GDSC -d NCI_IOA_AOA --ns 10 --nd 5 -m saved.uq.model.h5 -w saved.uq.weights.h5 -n 100

$ cat comb_pred_GDSC_NCI_IOA_AOA.tsv
Sample  Drug1   Drug2   N       PredGrowthMean  PredGrowthStd   PredGrowthMin   PredGrowthMax
GDSC.22RV1      NSC.102816      NSC.102816      100     0.1688  0.0899  -0.0762 0.3912
GDSC.22RV1      NSC.102816      NSC.105014      100     0.3189  0.0920  0.0914  0.5550
GDSC.22RV1      NSC.102816      NSC.109724      100     0.6514  0.0894  0.4739  0.9055
GDSC.22RV1      NSC.102816      NSC.118218      100     0.5682  0.1164  0.2273  0.8891
GDSC.22RV1      NSC.102816      NSC.122758      100     0.3787  0.0833  0.1779  0.5768
GDSC.22RV1      NSC.105014      NSC.102816      100     0.1627  0.1060  -0.0531 0.5077
...
```

A version of trained model files with dropout are available here: [saved.uq.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.uq.model.h5) and [saved.uq.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.uq.weights.h5).

## Profile runs
We have run the same configuration across multiple machines and compared the resource utilization. 
```
python uno_baseline_keras2.py --conf combo_perf_benchmark.txt
```

| Machine | Time to complete (HH:mm:ss) | Time per epoch (s) | Perf factor <sup>*</sup> | CPU % | Mem % | Mem GB | GPU % | GPU Mem % | Note |
| ------- | --------------------------: | -----------------: | -----------------------: | ----: | ----: | -----: | ----: | --------: | ---- |
| Theta | 1:14:12 | 811 | 0.31 | 7.6 | 7.6 | 12.8 | | |
| Nucleus | 0:14:13 | 72 | 3.47 | 3.8 | 9.3 | 21.9 | 63.4 | 91.9 |
| Tesla (K20) | 0:44:17 | 250 | 1.00 | 3.9 | 42.3 | 12.9 | 73.8 | 53.3 |
| Titan | | | | | | | | | keras version 2.0.3 does not supprot model.clone_model() which is introduced in 2.0.7 |
* Time per epoch on the machine divided by time per epoch of Titan (or Tesla)
