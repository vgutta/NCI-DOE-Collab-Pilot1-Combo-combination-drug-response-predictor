
import os, sys

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

from file_utils import get_file


modac_collection_path='https://modac.cancer.gov/api/v2/dataObject/NCI_DOE_Archive/JDACS4C/JDACS4C_Pilot_1/combination_drug_response_predictor'

model = 'combo.model.h5'
weights = 'combo.weights.h5'

model_url = os.path.join( modac_collection_path, model)
weights_url = os.path.join( modac_collection_path, weights)

get_file('uq.model.h5', model_url, datadir = '.')
get_file('uq.weights.h5', weights_url, datadir = '.')
