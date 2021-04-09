# NCI-DOE-Collab-Pilot1-Combo-Combination-Drug-Response-Predictor

### Description
The combination drug response predictor (Combo) shows how to predict tumor cell line growth to drug pairs in the [NCI-ALMANAC](https://www.ncbi.nlm.nih.gov/pubmed/28446463) database using artificial neural networks.

### User Community
Researchers interested in bioinformatics; computational cancer biology, drug discovery, and machine learning 

### Usability
Data scientists can train the provided untrained model with the shared preprocessed data or with their own preprocessed data, or can use the trained model to predict the drug response from the NCI-ALMANAC study. The provided scripts use data that have been downloaded from NCI-ALMANAC and normalized.

### Uniqueness
Using machine learning to predict drug response can be carried using multiple techniques. The general rule is that classical methods like random forests would perform better for small size datasets, while neural network approaches like Combo would perform better for relatively larger size data.

### Components
* Processed training and test data
* Untrained neural network model
* Trained model weights and topology to be used in inference

### Publication
Xia, Fangfang, et al. ["Predicting tumor cell line response to drug pairs with deep learning."](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2509-3?optIn=true) BMC bioinformatics 19.18 (2018): 71-79.

### Technical Details
Refer to this [README](./Pilot1/Combo/README.md).
