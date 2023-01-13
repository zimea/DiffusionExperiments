import os
import pandas as pd
import numpy as np
import glob
from Utility import timeit
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr


class DataReader:
    def __init__(self, config, prior_means, prior_stds):
        self.config = config
        self.prior_means = prior_means
        self.prior_stds = prior_stds

    def calculate_V(self, path: str):
        df = pd.read_csv(path, sep="\t")
        df = df[df[str(self.config.grid_size)] != self.config.grid_size]
        morpheus_ts = np.repeat(
            range(0, self.config.timesteps + 1), self.config.grid_size
        )
        times = np.tile(morpheus_ts, int(len(df.index) / morpheus_ts.shape[0]))
        if len(times) != len(df.index):
            print(path)
        df["time"] = np.tile(morpheus_ts, int(len(df.index) / morpheus_ts.shape[0]))
        df["sum_V"] = df.iloc[:, 1 : self.config.grid_size + 1].sum(axis=1)
        df = df.drop(columns=[str(x) for x in range(0, self.config.grid_size + 1)])
        df = df.groupby(list(df.columns[:-1])).agg({"sum_V": "sum"}).reset_index()
        return np.expand_dims(df["sum_V"], axis=1)[
            self.config.cut_off_start
            + 1 : self.config.timesteps
            - self.config.cut_off_end
        ]

    def calculate_volume(self, path: str):
        logger_state = "logger_1.csv"
        logger_volume = "logger_4_cell.id.csv"
        state = pd.read_csv(os.path.join(path, logger_state), sep="\t").rename(
            columns={"cell.id": "id"}
        )
        volume = np.genfromtxt(os.path.join(path, logger_volume), delimiter="\t")
        lines_per_timepoint = (self.config.timesteps + 1) * (self.config.grid_size + 1)
        volume = np.delete(
            volume,
            list(range(0, int(lines_per_timepoint), self.config.grid_size + 1)),
            axis=0,
        )

        first_ts_after_freeze = self.config.grid_size * (self.config.cut_off_start + 1)
        cell_id, counts = np.array(
            np.unique(
                np.array(volume)[
                    first_ts_after_freeze : first_ts_after_freeze
                    + self.config.grid_size,
                    1:,
                ],
                return_counts=True,
            )
        )
        counts = np.append(
            np.expand_dims(cell_id, axis=1), np.expand_dims(counts, axis=1), axis=1
        )
        if counts[0, 0] != 0:
            counts = np.insert(counts, 0, [[0, 0]], axis=0)
        counts = pd.DataFrame(counts, columns=["id", "count"])
        df = pd.merge(left=state, right=counts, how="left", on=["id"])
        df = df.groupby(["time", "V"]).agg({"count": "sum"}).reset_index()
        return np.expand_dims(df.query("V == 1")["count"], axis=1)

    @timeit
    def read_offline_data(self, path: str, workdir):
        with open(os.path.join(workdir, "log_read_offline.txt"), "w") as logfile:
            with redirect_stdout(logfile), redirect_stderr(logfile):
                path_list = glob.glob(path)
                nr_of_params = self.config.param_nr

                n_sim = len(path_list)
                dfs = np.empty(
                    (
                        n_sim,
                        (
                            self.config.timesteps
                            - 1
                            - self.config.cut_off_start
                            - self.config.cut_off_end
                        ),
                        4,
                    ),
                    dtype=np.float32,
                )
                params = np.empty((n_sim, nr_of_params), dtype=np.float32)
                invalidIndices = []

                for path in tqdm(range(n_sim)):
                    pathname = path_list[path]
                    filename = os.path.join(pathname, "logger_2.csv")
                    filename_V = os.path.join(pathname, "logger_6_Ve.csv")
                    df = pd.read_csv(filename, index_col=None, header=0, delimiter="\t")
                    path_split = filename.split("/")[len(filename.split("/")) - 2]
                    if "e" in path_split:
                        invalidIndices.append(path)
                        continue
                    if path_split.startswith("sweep") or path_split.startswith("DV"):
                        start_nr = 1
                    else:
                        start_nr = 0
                    params_split = path_split.split("_")[start_nr : nr_of_params + 1]
                    param_file = list(
                        map(lambda x: round(float(x.split("-")[1]), 3), params_split)
                    )

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
                    try:
                        df_V = self.calculate_V(filename_V)
                        I_volume = self.calculate_volume(pathname)[
                            self.config.cut_off_start
                            + 1 : self.config.timesteps
                            - self.config.cut_off_end
                        ]
                    except Exception as error:
                        invalidIndices.append(path)
                        continue

                    if (
                        np.any(df_inf < 1)
                        or np.any(np.asarray(param_file) > 1)
                        or len(df_inf)
                        != (
                            self.config.timesteps
                            - 1
                            - self.config.cut_off_start
                            - self.config.cut_off_end
                        )
                    ):
                        invalidIndices.append(path)
                        continue
                    params[path] = param_file
                    dfs[path] = np.append(
                        np.append(df_cells, df_V, axis=1), I_volume, axis=1
                    )

                dfs = np.delete(dfs, invalidIndices, axis=0)
                params = np.delete(params, invalidIndices, axis=0)
                print("Read data in the form of: ", dfs.shape)

                return dfs, params

    @timeit
    def read_offline_data_2d(self, path: str, workdir):
        with open(os.path.join(workdir, "log_read_offline.txt"), "w") as logfile:
            with redirect_stdout(logfile), redirect_stderr(logfile):
                path_list = glob.glob(path)
                nr_of_params = self.config.param_nr

                n_sim = len(path_list)
                dfs = np.empty(
                    (
                        n_sim,
                        self.config.grid_size,
                        self.config.grid_size,
                        (
                            self.config.timesteps
                            - self.config.cut_off_end
                            - self.config.cut_off_start
                            - 1
                        ),  # TODO: Zeit kürzen
                        3,
                    ),
                    dtype=np.float32,
                )
                params = np.empty((n_sim, nr_of_params), dtype=np.float32)
                invalidIndices = []

                for path in tqdm(range(n_sim)):
                    pathname = path_list[path]
                    path_split = pathname.split("/")[len(pathname.split("/")) - 1]
                    if "e" in path_split:
                        invalidIndices.append(path)
                        continue

                    params_dict = {
                        i.split("-")[0]: round(float(i.split("-")[1]), 3)
                        for i in path_split.split("_")
                        if i.split("-")[0] in self.config.prior_names
                    }

                    v = self.calculate_V_2d(pathname).transpose((1, 2, 0))[
                        :,
                        :,
                        self.config.cut_off_start
                        + 1 : self.config.timesteps
                        - self.config.cut_off_end,
                    ]
                    I = self.get_cell_states_2d(pathname).transpose((1, 2, 0))[
                        :,
                        :,
                        self.config.cut_off_start
                        + 1 : self.config.timesteps
                        - self.config.cut_off_end,
                    ]
                    id = self.get_ids_2d(pathname).transpose((1, 2, 0))[
                        :,
                        :,
                        self.config.cut_off_start
                        + 1 : self.config.timesteps
                        - self.config.cut_off_end,
                    ]
                    params[path] = list(params_dict.values())
                    dfs[path] = np.concatenate(
                        (
                            np.expand_dims(v, axis=-1),
                            np.expand_dims(id, axis=-1),
                            np.expand_dims(I, axis=-1),
                        ),
                        axis=-1,
                    )

                dfs = np.delete(dfs, invalidIndices, axis=0)
                params = np.delete(params, invalidIndices, axis=0)
                print("Read data in the form of: ", dfs.shape)
                return dfs, params

    def prepare_input(self, forward_dict):
        """Function to self.configure the simulated quantities (i.e., simulator outputs)
        into a neural network-friendly (BayesFlow) format.
        """

        # Prepare placeholder dict
        out_dict = {}

        # Convert data to logscale

        logdata = np.log1p(forward_dict["sim_data"]).astype(np.float64)

        # Extract prior draws and z-standardize with previously computed means
        params = forward_dict["prior_draws"].astype(np.float64)
        params = (params - self.prior_means) / self.prior_stds

        # Remove a batch if it contains nan, inf or -inf
        # idx_keep = np.all(np.isfinite(logdata), axis=(1, 2))

        # Add to keys
        out_dict["summary_conditions"] = logdata
        out_dict["parameters"] = params
        return out_dict

    def __get_2d_data__(self, path: str, logger: str, to_np: bool = True):
        logger_data = os.path.join(path, logger)
        df = pd.read_csv(logger_data, sep="\t")
        df = df[df[str(self.config.grid_size)] != self.config.grid_size].drop(
            columns=str(self.config.grid_size)
        )
        if to_np:
            df = df.to_numpy().reshape(
                self.config.timesteps + 1, self.config.grid_size, self.config.grid_size
            )
        return df

    def calculate_V_2d(self, path: str, to_np: bool = True):
        Ve = self.__get_2d_data__(path, "logger_6_Ve.csv", to_np=to_np)
        return Ve

    def get_ids_2d(self, path: str, to_np: bool = True):
        Ids = self.__get_2d_data__(path, "logger_4_cell.id.csv", to_np=to_np)
        return Ids

    def get_cell_states(self, path: str, to_np: bool = True):
        cell_state = pd.read_csv(os.path.join(path, "logger_1.csv"), sep="\t")
        if to_np:
            cell_state = (
                cell_state.drop(columns="time")
                .to_numpy()
                .reshape(self.config.timesteps, self.config.cell_nr, 2)
            )
        return cell_state

    def get_cell_states_2d(self, path: str, to_np: bool = True):
        id = self.get_ids_2d(path, False)
        time = np.repeat(range(0, self.config.cell_nr), self.config.grid_size)
        id["time"] = time
        cs = []

        cell_state = self.get_cell_states(path, False)
        for t in id["time"].unique():
            dict = (
                cell_state[cell_state["time"] == t]
                .drop(columns="time")
                .set_index("cell.id")
                .to_dict()["V"]
            )

            cs.append(id[id["time"] == t].drop(columns="time").replace(dict))

        cs = pd.concat(cs)
        if to_np:
            cs = cs.to_numpy().reshape(
                self.config.timesteps + 1, self.config.grid_size, self.config.grid_size
            )
        return cs
