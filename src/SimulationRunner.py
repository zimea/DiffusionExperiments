import glob
import os, sys
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import time
from pandas import read_csv
from Utility import timeit
from contextlib import redirect_stdout, redirect_stderr
from SimulationRunnerInterface import SimulationRunnerInterface


class SimulationRunner(SimulationRunnerInterface):
    def __init__(self, config, workdir, dataReader):
        SimulationRunnerInterface.__init__(self)
        self.config = config
        self.workdir = workdir
        self.dataReader = dataReader

    @timeit
    def run(self, params):
        with open(os.path.join(self.workdir, "log_morpheus.txt"), "w") as logfile:
            with redirect_stderr(logfile), redirect_stdout(logfile):
                model_dir = "model"
                model_pattern = os.path.join(
                    self.config.data_path,
                    model_dir,
                    self.config.model_pattern,
                )
                models = glob.glob(model_pattern)
                model = models[0]

                priors = dict(zip(self.config.prior_names, params))
                path_param_str = "_".join(
                    "-".join((key, str(value))) for key, value in priors.items()
                )
                morpheus_param_str = " ".join(
                    "=".join(("-" + key, str(value))) for key, value in priors.items()
                )

                if len(self.config.fixed_params) > 0:
                    path_param_str += "_" + "_".join(
                        "-".join((key, str(value)))
                        for key, value in self.config.fixed_params.items()
                    )
                    morpheus_param_str += " " + " ".join(
                        "=".join(("-" + key, str(value)))
                        for key, value in self.config.fixed_params.items()
                    )

                OUT = os.path.join(
                    self.config.data_path, self.config.folder, path_param_str
                )
                create_dir = Popen(
                    "mkdir " + OUT, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                create_dir.wait()
                morpheus_command = (
                    "morpheus"
                    + " -f "
                    + model
                    + " -o "
                    + OUT
                    + " "
                    + morpheus_param_str
                )
                print(morpheus_command)

                run_sim = Popen(
                    morpheus_command, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                run_sim.wait()

                final_plot = os.path.join(
                    OUT, "plot_" + str(self.config.timesteps).zfill(5) + ".png"
                )
                while not os.path.exists(final_plot):
                    time.sleep(1)

                population_file = os.path.join(OUT, "logger_2.csv")
                df = read_csv(population_file, sep="\t")
                df_tar = df["celltype.target.size"].values[:, np.newaxis][
                    self.config.cut_off_start
                    + 1 : self.config.timesteps
                    - self.config.cut_off_end
                ]
                df_inf = df["celltype.infected.size"].values[:, np.newaxis][
                    self.config.cut_off_start
                    + 1 : self.config.timesteps
                    - self.config.cut_off_end
                ]
                df_cells = np.append(df_tar, df_inf, axis=1)

                v_path = os.path.join(OUT, "logger_6_Ve.csv")
                v = self.dataReader.calculate_V(v_path)
                sim = np.append(df_cells, v, axis=1)
                I_volume = self.dataReader.calculate_volume(OUT)[
                    self.config.cut_off_start
                    + 1 : self.config.timesteps
                    - self.config.cut_off_end
                ]
                sim = np.append(sim, I_volume, axis=1)

                return sim

    @timeit
    def run_2d(self, params):
        with open(os.path.join(self.workdir, "log_morpheus.txt"), "w") as logfile:
            with redirect_stderr(logfile), redirect_stdout(logfile):
                model_dir = "model"
                model_pattern = os.path.join(
                    self.config.data_path,
                    model_dir,
                    self.config.model_pattern,
                )
                models = glob.glob(model_pattern)
                model = models[0]

                priors = dict(zip(self.config.prior_names, params))
                path_param_str = "_".join(
                    "-".join((key, str(value))) for key, value in priors.items()
                )
                morpheus_param_str = " ".join(
                    "=".join(("-" + key, str(value))) for key, value in priors.items()
                )

                if len(self.config.fixed_params) > 0:
                    path_param_str += "_" + "_".join(
                        "-".join((key, str(value)))
                        for key, value in self.config.fixed_params.items()
                    )
                    morpheus_param_str += " " + " ".join(
                        "=".join(("-" + key, str(value)))
                        for key, value in self.config.fixed_params.items()
                    )

                OUT = os.path.join(
                    self.config.data_path, self.config.folder, path_param_str
                )
                create_dir = Popen(
                    "mkdir " + OUT, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                create_dir.wait()
                morpheus_command = (
                    "morpheus"
                    + " -f "
                    + model
                    + " -o "
                    + OUT
                    + " "
                    + morpheus_param_str
                )
                print(morpheus_command)

                run_sim = Popen(
                    morpheus_command, shell=True, stdout=sys.stdout, stderr=sys.stderr
                )
                run_sim.wait()

                final_plot = os.path.join(
                    OUT, "plot_" + str(self.config.timesteps).zfill(5) + ".png"
                )
                while not os.path.exists(final_plot):
                    time.sleep(1)

                v = self.dataReader.calculate_V_2d(OUT).transpose((1, 2, 0))[
                    :,
                    :,
                    self.config.cut_off_start
                    + 1 : self.config.timesteps
                    - self.config.cut_off_end,
                ]
                I = self.dataReader.get_cell_states_2d(OUT).transpose((1, 2, 0))[
                    :,
                    :,
                    self.config.cut_off_start
                    + 1 : self.config.timesteps
                    - self.config.cut_off_end,
                ]
                id = self.dataReader.get_ids_2d(OUT).transpose((1, 2, 0))[
                    :,
                    :,
                    self.config.cut_off_start
                    + 1 : self.config.timesteps
                    - self.config.cut_off_end,
                ]
                sim = np.concatenate(
                    (
                        np.expand_dims(v, axis=-1),
                        np.expand_dims(id, axis=-1),
                        np.expand_dims(I, axis=-1),
                    ),
                    axis=-1,
                )

                return sim
