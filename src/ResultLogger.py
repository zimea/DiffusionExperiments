import os
from matplotlib import pyplot as plt
import numpy as np
from SimulationRunnerInterface import SimulationRunnerInterface
import pandas as pd
import plotnine as p9


class ResultLogger:
    def __init__(
        self,
        workdir,
        trainer,
        losses,
        config,
        prior,
        diag,
        model,
        configurator,
        amortizer,
    ):
        self.workdir = workdir
        self.trainer = trainer
        self.losses = losses
        self.config = config
        self.prior = prior
        self.diag = diag
        self.model = model
        self.configurator = configurator
        self.amortizer = amortizer

        self.output_dir = os.path.abspath(os.path.join(workdir, config.plots))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.prior_means, self.prior_stds = prior.estimate_means_and_stds()

    def __percentile__(self, n):
        def percentile(x):
            return np.percentile(x, n)

        percentile.__name__ = "percentile_%s" % n
        return percentile

    def __get_resimulation_data__(self):
        res = {}
        res["raw_sims"] = self.model(
            batch_size=self.config.resimulation_param["simulations"]
        )
        res["validation_sims"] = self.configurator(res["raw_sims"])
        res["post_samples"] = self.amortizer.sample(
            res["validation_sims"], self.config.resimulation_param["post_samples"]
        )
        res["post_samples_unnorm"] = (
            self.prior_means + res["post_samples"] * self.prior_stds
        )
        return res

    def create_plots(self):
        plot_dir = os.path.abspath(os.path.join(self.workdir, self.config.plots))

        if self.config.losses:
            loss = self.diag.plot_losses(self.losses)
            loss.savefig(os.path.join(plot_dir, "losses.png"))
            plt.close(loss)

        if self.config.latent2d:
            latent2d = self.trainer.diagnose_latent2d()
            latent2d.savefig(os.path.join(plot_dir, "latent2d.png"))
            plt.close(latent2d)

        if self.config.sbc_histograms:
            sbc_histograms = self.trainer.diagnose_sbc_histograms()
            sbc_histograms.savefig(os.path.join(plot_dir, "sbc_histograms.png"))
            plt.close(sbc_histograms)

        if self.config.run_resimualtions:
            res = self.__get_resimulation_data__()

        if self.config.sbc_ecdf:
            sbc_ecdf = self.diag.plot_sbc_ecdf(
                res["post_samples"], res["validation_sims"]["parameters"]
            )
            sbc_ecdf.savefig(os.path.join(plot_dir, "sbc_ecdf.png"))
            plt.close(sbc_ecdf)

        # TODO: posterior scores, correlation

        if self.config.recovery:
            recovery = self.diag.plot_recovery(
                res["post_samples"],
                res["validation_sims"]["parameters"],
                param_names=self.config.prior_names,
            )
            recovery.savefig(os.path.join(plot_dir, "recovery.png"))
            plt.close(recovery)

        # if self.config.post_eval:
        #     fig, plot = self.plot_posterior_eval(res)
        #     fig.savefig(os.path.join(plot_dir, "post_eval.png"))

        # if self.config.resimulation:
        #     for rep in range(self.config.nr_resimulations):
        #         fig, plot = self.create_resimulation_plot(
        #             simulationRunner, res["post_samples"][rep], res["raw_sims"][""]
        #         )

    def create_resimulation_plot(self, simulation, posterior, ground_truth, id):
        simulationRunner: SimulationRunnerInterface = simulation

        cut_ts = (
            self.config.timesteps
            - self.config.cut_off_end
            - self.config.cut_off_start
            - 1
        )
        time_range = range(0, cut_ts)
        nr_resimulations = 10

        resim = np.zeros(
            ((nr_resimulations * cut_ts, len(self.config.observables) + 1))
        )
        for i in range(nr_resimulations):
            resim[i * cut_ts : (i + 1) * cut_ts, :] = np.append(
                np.asarray(simulationRunner.run(posterior)),
                np.expand_dims(range(0, cut_ts), axis=1),
            )
        df = pd.DataFrame(resim, columns=self.config.observables.append("time"))
        agg = df.groupby(["time"]).agg(
            [
                "mean",
                "std",
                self.__percentile__(0.25),
                self.__percentile__(0.75),
                self.__percentile__(0.05),
                self.__percentile__(0.95),
                self.__percentile__(0.025),
                self.__percentile__(0.975),
            ]
        )
        gt_index = pd.MultiIndex.from_arrays(
            [self.config.observables, np.repeat("gt", len(self.config.observables))]
        )
        gt = pd.DataFrame(ground_truth)
        agg[gt_index] = gt
        agg = agg.sort_index(axis=1)
        agg = (
            pd.melt(agg.reset_index(), id_vars=["t"])
            .pivot(index=["t", "variable_0"], columns="variable_1")["value"]
            .reset_index()
        )

        fig, plot = (
            p9.ggplot(agg, p9.aes(x="t"))
            + p9.geom_ribbon(
                p9.aes(ymin="percentile_0.25", ymax="percentile_0.75"),
                fill="blue",
                alpha=0.2,
            )
            + p9.geom_ribbon(
                p9.aes(ymin="percentile_0.05", ymax="percentile_0.95"),
                fill="blue",
                alpha=0.3,
            )
            + p9.geom_ribbon(
                p9.aes(ymin="percentile_0.025", ymax="percentile_0.975"),
                fill="blue",
                alpha=0.4,
            )
            + p9.geom_line(p9.aes(y="mean"), color="black")
            + p9.geom_point(p9.aes(y="mean"))
            + p9.facet_grid("variable_0 ~ .")
        )
        return fig, plot

    def plot_posterior_eval(self, resimulation_data):
        df_prior = pd.DataFrame(
            resimulation_data["raw_sims"]["prior_draws"],
            columns=self.config.prior_names,
        )
        df_prior = pd.melt(
            df_prior, value_vars=self.config.prior_names, value_name="prior"
        )

        post_mean = np.mean(resimulation_data["post_samples_unnorm"], axis=1)
        post_sd = np.std(resimulation_data["post_samples_unnorm"], axis=1)
        df_prior["z_score"] = np.array(
            (post_mean - resimulation_data["raw_sims"]["prior_draws"]) / post_sd
        ).flatten()
        df_prior["contraction"] = np.array(
            1 - (post_sd / self.prior_stds) ** 2
        ).flatten()

        fig, plot = (
            p9.ggplot(df_prior, p9.aes(x="contraction", y="z_score", color="prior"))
            + p9.geom_point()
            + p9.facet_grid("variable ~ .")
        )
        fig.savefig(os.path.join(self.plot_dir, "posterior_eval.png"))
