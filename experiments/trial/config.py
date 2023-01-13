# config uses a python file, which is questionable in production use. Instead e.g. yaml would be a better choice

import sys, os

SourcePath = "/home/l/projects/Morpheus/Tutorial/DiffusionExperiments/src"
BayesFlowPath = "/home/l/projects/Morpheus/Tutorial/infectionSpread/BayesFlow"
sys.path.append(os.path.abspath(SourcePath))
import PriorFunctions
import SummaryNetworks

# set BayesFlow modules
prior_names = [r"DV", r"bcf"]
prior_func = PriorFunctions.prior_DV
fixed_params = {"cV": 0.5, "pV": 0.5}
summary_parm = 16
summary_network = SummaryNetworks.ConvLSTM(n_summary=summary_parm)

# where to put the results
resultsPath = "experiments"
checkpoints = "checkpoints"
plots = "plots"

# data config
data_path = "/home/l/projects/Morpheus/Modelle/cell_free_50_diff"
folder = "output_DV_bcf_trial"
model_pattern = "cell_free_50_DV_bcf.xml"
cell_nr = 51
timesteps = 50
cut_off_start = 9
cut_off_end = 10
spatial = True

# training hyperparameter
param_nr = len(prior_func())
inn_layer = 4
batch_size = 8
iter_per_epoch = 20
epochs = 5
grid_size = 10
retrain = False
model_name = "morpheus"
training_mode = "offline"
amortizer_name = "emune_amortizer"
optional_stopping = True

# which plots and diagnostics
losses = True
latent2d = True
sbc_histograms = True
sbc_ecdf = True
posterior_scores = True
recovery = True
correlation = True
slope = True
post_eval = True
resimulation = True
nr_resimulations = 3
run_resimualtions = (
    sbc_ecdf
    or posterior_scores
    or recovery
    or correlation
    or slope
    or post_eval
    or resimulation
)
resimulation_param = {"simulations": 100, "post_samples": 500}
