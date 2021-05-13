## Combo: Predicting Tumor Cell Line Response to Drug Pairs

### Overview
Given combination drug screening results on NCI-60 cell lines available at the [NCI-ALMANAC](https://www.ncbi.nlm.nih.gov/pubmed/28446463) database, this deep learning network can predict the growth percentage from the cell line molecular features and the descriptors of both drugs.

**Relationship to core problem**: This benchmark is an example one of the core capabilities developed for the Pilot 1 Drug Response problem: combining multiple molecular assays and drug descriptors in a single deep learning framework for response prediction.

**Outcome**: This Deep Neural Network can predict growth percentage of a cell line treated with a pair of drugs.


#### Description of the Data
* Data sources: 
   * Combo drug response screening results from NCI-ALMANAC; 
   * 5-platform normalized expression, microRNA expression, and proteome abundance data from the NCI; 
   * Dragon7 generated drug descriptors based on 2D chemical structures from NCI.
* Input dimensions: 
   * ~30K with default options: 26K normalized expression levels by gene and 4K drug descriptors; 
   * 59 cell lines; 
   * a subset of 54 FDA-approved drugs.
* Output dimensions: 1 (growth percentage).
* Sample size: 85,303 tuples (cell line, drug 1, drug 2) from the original 304,549 in the NCI-ALMANAC database.
* Notes on data balance: Ineffective drug pairs exceed effective pairs; data imbalance is somewhat reduced by using only the best dose combination for each tuple (cell line, drug 1, drug 2) as training and validation data.


#### Outcomes
* Regression: Predict percent growth for any NCI-60 cell line and drugs combination.
* Dimension: 1 scalar value corresponding to the percent growth for a given drug concentration. 
* Output range: [-100, 100].

#### Evaluation Metrics
* Accuracy or loss function: Mean squared error, mean absolute error, and R^2.
* Performance of a naïve method: Mean response, linear regression, or random forest regression.

#### Description of the Network
* Network architecture: Two-stage neural network that is jointly trained for feature encoding and response prediction; shared submodel for each drug in the pair.
* Number of layers: 3 layers for feature encoding submodels and 4 layers for response prediction submodels. 


### Setup
To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n Combo
   conda activate Combo
   ```

### Running the Baseline Implementation

To run the baseline implementation, execute the following command. For a detailed explanation of command line arguments, refer to [combo.py](./combo.py).

```
$ cd Pilot1/Combo
$ python combo_baseline_keras2.py --cell_features rnaseq --drug_features descriptors --residual True --cp True --epochs 100 --use_landmark_genes True --warmup_lr True --reduce_lr True --base_lr 0.0003 -z 128 --preprocess_rna source_scale --dropout 0.2 --save_path save/uq
```

#### Example Output
```
Loaded 317899 unique (CL, D1, D2) response sets.
Filtered down to 276112 rows with matching information.
Unique cell lines: 60
Unique drugs: 98
Distribution of dose response:
              GROWTH
count  276112.000000
mean        0.334128
std         0.526155
min        -1.000000
25%         0.059500
50%         0.420100
75%         0.780253
max         1.693300
Rows in train: 220890, val: 55222
Input features shapes:
  cell.rnaseq: (942,)
  drug1.descriptors: (3839,)
  drug2.descriptors: (3839,)
Total input dimensions: 8620

Between random pairs in y_val:
  mse: 0.5537
  mae: 0.5831
  r2: -1.0010
  corr: -0.0005
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 942)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 1000)              943000    
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              1001000   
_________________________________________________________________
dense_3 (Dense)              (None, 1000)              1001000   
=================================================================
Total params: 2,945,000
Trainable params: 2,945,000
Non-trainable params: 0
_________________________________________________________________
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 3839)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 1000)              3840000   
_________________________________________________________________
dense_5 (Dense)              (None, 1000)              1001000   
_________________________________________________________________
dense_6 (Dense)              (None, 1000)              1001000   
=================================================================
Total params: 5,842,000
Trainable params: 5,842,000
Non-trainable params: 0
_________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input.cell.rnaseq (InputLayer)  (None, 942)          0                                            
__________________________________________________________________________________________________
input.drug1.descriptors (InputL (None, 3839)         0                                            
__________________________________________________________________________________________________
input.drug2.descriptors (InputL (None, 3839)         0                                            
__________________________________________________________________________________________________
cell.rnaseq (Model)             (None, 1000)         2945000     input.cell.rnaseq[0][0]          
__________________________________________________________________________________________________
drug.descriptors (Model)        (None, 1000)         5842000     input.drug1.descriptors[0][0]    
                                                                 input.drug2.descriptors[0][0]    
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 3000)         0           cell.rnaseq[1][0]                
                                                                 drug.descriptors[1][0]           
                                                                 drug.descriptors[2][0]           
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1000)         3001000     concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 1000)         1001000     dense_7[0][0]                    
__________________________________________________________________________________________________
add_2 (Add)                     (None, 1000)         0           dense_8[0][0]                    
                                                                 dense_7[0][0]                    
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1000)         1001000     add_2[0][0]                      
__________________________________________________________________________________________________
add_3 (Add)                     (None, 1000)         0           dense_9[0][0]                    
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            1001        add_3[0][0]                      
==================================================================================================
Total params: 13,791,001
Trainable params: 13,791,001
Non-trainable params: 0
__________________________________________________________________________________________________
Train on 220890 samples, validate on 55222 samples
...
...

Epoch 100/100
220890/220890 [==============================] - 22s 102us/step - loss: 0.0110 - mae: 0.0724 - r2: 0.9595 - val_loss: 0.0224 - val_mae: 0.0970 - val_r2: 0.9172
Comparing y_true and y_pred:
  mse: 0.0225
  mae: 0.0972
  r2: 0.9187
  corr: 0.9585
```

#### Inference

There is a separate inference script that can be used to predict drug pair response on combinations of sample sets and drug sets with a trained model.

A version of trained model files with dropout are available on [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7678072).

```
#small inference for testing
$python download_model.py
$python infer.py --sample_set NCIPDM --drug_set ALMANAC --use_landmark_genes -m uq.model.h5 -w uq.weights.h5

Using TensorFlow backend.
Predicting drug response for 6381440 combinations: 590 samples x 104 drugs x 104 drugs
100%|██████████████████████████████████████████████████████████████████████| 639/639 [14:56<00:00,  1.40s/it]
```

The inference script also accepts models trained with [dropout as a Bayesian Approximation](https://arxiv.org/pdf/1506.02142.pdf) for uncertainty quantification. 

Combo inference generates two files:
* comb_pred_{cellset}_{drugset}.all.tsv has all prediction instances: Sample, Drug1, Drug2, N, Seq, PredGrowth
* comb_pred_{cellset}_{drugset}.all.tsv contains the aggregated statistics: Sample, Drug1, Drug2, N, PredGrowthMean, PredGrowthStd, PredGrowthMin, PredGrowthMax

Note that the inference code can be used to generate multiple predictions for the same sample-drug pair with the `--n_pred` parameter. This number is shown in the N column, and the sequential number for the individual predictions is denoted by Seq.

Here is an example command line to make 10 point predictions for each sample-drug combination in a subsample of the Genomics of Drug Sensitivity in Cancer (GDSC) data.


```
$python download_model.py
$python infer.py -s GDSC -d NCI_IOA_AOA --ns 10 --nd 5 --use_landmark_genes -m uq.model.h5 -w uq.weights.h5 -n 10

$head comb_pred_GDSC_NCI_IOA_AOA.tsv

Sample  Drug1   Drug2   N   PredGrowthMean  PredGrowthStd   PredGrowthMin   PredGrowthMax
GDSC.22RV1  NSC.102816  NSC.102816  10  0.7323  0.1722  0.3545  0.8919
GDSC.22RV1  NSC.102816  NSC.105014  10  0.7638  0.2280  0.1091  0.9200
GDSC.22RV1  NSC.102816  NSC.109724  10  0.8755  0.0491  0.7893  0.9542
GDSC.22RV1  NSC.102816  NSC.118218  10  0.8710  0.0424  0.7593  0.9130
GDSC.22RV1  NSC.102816  NSC.122758  10  0.8558  0.0509  0.7437  0.9263
GDSC.22RV1  NSC.105014  NSC.102816  10  0.7994  0.1995  0.2616  0.9429
GDSC.22RV1  NSC.105014  NSC.105014  10  0.8662  0.0598  0.7160  0.9248
GDSC.22RV1  NSC.105014  NSC.109724  10  0.8024  0.1276  0.4626  0.9312
GDSC.22RV1  NSC.105014  NSC.118218  10  0.8673  0.1094  0.5828  1.0078
...
```




