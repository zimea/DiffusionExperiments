import config, sys, os
from DataReader import DataReader
import PriorFunctions

sys.path.append(os.path.abspath(os.path.join(config.BayesFlowPath)))
from bayesflow.forward_inference import Prior

if __name__ == "__main__":
    prior_func = PriorFunctions.prior_DV
    prior = Prior(prior_fun=prior_func, param_names=config.prior_names)
    prior_means, prior_stds = prior.estimate_means_and_stds()
    dataReader = DataReader(config, prior_means=prior_means, prior_stds=prior_stds)
    df = dataReader.get_cell_states_2d(
        path="/home/l/projects/Morpheus/Modelle/cell_free_50_diff/output_DV_bcf/DV-0.3953700819631_bcf-0.12939422940223888_cV-0.5_pV-0.5",
        to_np=False,
    )
    print(df)
