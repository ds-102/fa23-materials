import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.animation as ani
from matplotlib.widgets import Button
import functools

sns.set()


class BanditEnv:

    def __init__(self, num_arms, means='rand', standard_deviations=None):
        """Creates a bandit environment with normal reward distributions.

        Args:
            num_arms: number of arms in the simulation
            means: either a list/array with the mean of each arm,
                or 'rand' to generate means from Unif(0, 10)
                and SDs from Unif(1.2, 2.2)
            standard_deviations: if means is an array, used as
                the SDs for each arm
        """
        self.num_arms = num_arms

        # if means is None:
        #     self.num_arms = num_arms
        #     self.means = list(10 * np.random.random((num_arms - 1)))
        #     self.means.insert(0, 10)
        #     self.standard_deviations = list(np.random.random((num_arms - 1)) + 1.2)
        #     self.standard_deviations.insert(0, 1.2)
        if means == "rand":
            self.means = np.random.uniform(0, 10, num_arms)
            self.standard_deviations = np.random.uniform(1.2, 2.2, num_arms)
        else:
            assert len(means) == len(standard_deviations)
            self.num_arms = len(means)
            self.means = means
            self.standard_deviations = standard_deviations

        self.COLORS = sns.color_palette("colorblind", self.num_arms)
        self.COLORS2 = sns.color_palette("Paired")

    def pull_arm(self, arm):

        mu = self.means[arm]
        sigma = self.standard_deviations[arm]

        return np.random.normal(mu, sigma**2)

    def make_background(self, show_dists=True, show_regret=True):
        if show_regret:
            fig, axs = plt.subplots(
                2, 1, gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1]}
            )
            fig.suptitle("UCB Demo")
            arms_ax, regret_ax = axs
        else:
            fig, arms_ax = plt.subplots(1, 1)
            fig.suptitle("UCB Demo")
            regret_ax = None

        if show_dists:
            for arm in range(self.num_arms):
                mu = self.means[arm]
                sigma = self.standard_deviations[arm]
                rect = patches.Rectangle((0, -10), 1, 25, facecolor="w", zorder=0)
                arms_ax.add_patch(rect)
                sns.kdeplot(
                    #data=np.random.normal(mu, sigma**2, (100000)),
                    data=None,
                    y=np.random.normal(mu, sigma**2, (100000)),
                    color=self.COLORS[arm],
                    alpha=1.0,
                    ax=arms_ax,
                )
                separation = 1.0 / self.num_arms
                arms_ax.plot(
                    [-separation * (self.num_arms + 0.5), 0],
                    [mu, mu],
                    ":",
                    lw=3,
                    c=self.COLORS[arm],
                )
            arms_ax.set_xlim([-separation * (self.num_arms + 0.5), 0.35])
        else:
            separation = 1.0 / self.num_arms
            arms_ax.set_xlim([-separation * (self.num_arms + 0.5), 0])

        return fig, arms_ax, regret_ax


class UCB:
    def __init__(self, bandit_env, delta):
        self.bandit_env = bandit_env
        self.num_arms = self.bandit_env.num_arms
        self.delta = delta

        self.times_pulled = None
        self.rewards = None
        self.upper_confidence_bounds = None

        self.figure = None

    def initialize(self, samples=None):
        self.times_pulled = [1 for arm in range(self.num_arms)]
        self.rewards = [[] for arm in range(self.num_arms)]
        self.upper_confidence_bounds = [0 for arm in range(self.num_arms)]

        if samples is None:
            for arm in range(self.num_arms):
                reward = self.bandit_env.pull_arm(arm)
                self.rewards[arm].append(reward)
                self.upper_confidence_bounds[arm] = reward + np.sqrt(
                    2 * np.log(1 / self.delta) / self.times_pulled[arm]
                )

        return

    def choose_arm_and_update_bounds(self, event, show_dists=1, show_regret=1):
        arm = np.argmax(self.upper_confidence_bounds)
        reward = self.bandit_env.pull_arm(arm)
        self.times_pulled[arm] += 1
        self.rewards[arm].append(reward)
        self.pseudo_regret += np.max(self.bandit_env.means) - self.bandit_env.means[arm]
        self.regret.append(self.pseudo_regret)

        sample_mean = np.mean(self.rewards[arm])
        self.upper_confidence_bounds[arm] = sample_mean + np.sqrt(
            2 * 2.2**2 * np.log(2 / self.delta) / self.times_pulled[arm]
        )

        self.viewBounds(show_dists, show_regret)
        if show_regret:
            self.regret_ax.plot(self.regret, "-.", c=self.bandit_env.COLORS2[1])
            self.regret_ax.set_xlabel("Time (t)", fontsize=10)
            self.regret_ax.set_ylabel("Pseudo-regret", fontsize=10)
        plt.draw()
        return arm, reward

    def interactive_choose_arm_and_update_bounds(
        self, event, pull, show_dists=True, show_regret=True
    ):
        arm = pull
        reward = self.bandit_env.pull_arm(arm)
        self.times_pulled[arm] += 1
        self.rewards[arm].append(reward)
        self.pseudo_regret += np.max(self.bandit_env.means) - self.bandit_env.means[arm]
        self.regret.append(self.pseudo_regret)

        sample_mean = np.mean(self.rewards[arm])
        self.upper_confidence_bounds[arm] = sample_mean + np.sqrt(
            2 * 2.2**2 * np.log(2 / self.delta) / self.times_pulled[arm]
        )

        self.viewBounds(show_dists, show_regret)
        if show_regret:
            self.regret_ax.plot(self.regret, "-.", c=self.bandit_env.COLORS2[1])
            self.regret_ax.set_xlabel("Time (t)", fontsize=10)
            self.regret_ax.set_ylabel("Pseudo-regret", fontsize=10)
        plt.draw()

        return arm, reward

    def run_Interactive(self, dists=1, reg=1, show_UCB=1):
        self.initialize()
        self.pseudo_regret = 0
        self.regret = []
        self.viewBounds(dists, reg)
        self.regret.append(self.pseudo_regret)
        if reg:
            self.regret_ax.plot(self.regret, "-.", c=self.bandit_env.COLORS2[1])
            self.regret_ax.set_xlabel("Time (t)", fontsize=10)
            self.regret_ax.set_ylabel("Pseudo-regret", fontsize=10)

        self._make_buttons(dists, reg, show_UCB)

        plt.show()

        return self.regret

    def _make_buttons(self, dists=1, reg=1, show_UCB=1):
        self.buttons = []
        textstarts = []
        separation = 1.0 / self.num_arms
        axes_height = 2.5
        axes_width = 0.5 * separation
        width, h = self.figure.transFigure.inverted().transform(
            self.arms_ax.transData.transform([axes_width, 0])
        ) - self.figure.transFigure.inverted().transform(
            self.arms_ax.transData.transform([0, 0])
        )

        w, height = self.figure.transFigure.inverted().transform(
            self.arms_ax.transData.transform([0, axes_height])
        ) - self.figure.transFigure.inverted().transform(
            self.arms_ax.transData.transform([0, 0])
        )

        for arm in range(self.num_arms):
            location = -separation * (self.num_arms - arm)
            t1, t2 = self.figure.transFigure.inverted().transform(
                self.arms_ax.transData.transform([location - axes_width / 2.0, -8.5])
            )
            textstarts.append([t1, t2, width, height])

        for arm in range(self.num_arms):
            self.buttons.append(
                Button(
                    plt.axes(textstarts[arm]),
                    str(arm),
                    color=self.bandit_env.COLORS[arm],
                    hovercolor=self.bandit_env.COLORS2[1],
                )
            )

            self.buttons[-1].on_clicked(
                functools.partial(
                    self.interactive_choose_arm_and_update_bounds,
                    pull=arm,
                    show_dists=dists,
                    show_regret=reg,
                )
            )

        if show_UCB:
            if dists:
                t1, t2 = self.figure.transFigure.inverted().transform(
                    self.arms_ax.transData.transform([0.1, -8.5])
                )
                textstarts.append([t1, t2, width, height])

                self.buttons.append(
                    Button(
                        plt.axes([t1, t2, 2 * width, height]),
                        "UCB",
                        color=self.bandit_env.COLORS2[0],
                        hovercolor=self.bandit_env.COLORS2[1],
                    )
                )

                self.buttons[-1].on_clicked(
                    functools.partial(
                        self.choose_arm_and_update_bounds,
                        show_dists=dists,
                        show_regret=reg,
                    )
                )

            else:
                t1, t2 = self.figure.transFigure.inverted().transform(
                    self.arms_ax.transData.transform([-0.5 - 2 * axes_width, -12.5])
                )
                textstarts.append([t1, t2, width, height])

                self.buttons.append(
                    Button(
                        plt.axes([t1, t2, 2 * width, height]),
                        "UCB",
                        color=self.bandit_env.COLORS2[0],
                        hovercolor=self.bandit_env.COLORS2[1],
                    )
                )

                self.buttons[-1].on_clicked(
                    functools.partial(
                        self.choose_arm_and_update_bounds,
                        show_dists=dists,
                        show_regret=reg,
                    )
                )

    def viewBounds(self, show_dists=1, show_regret=1):
        if self.figure is None:
            self.figure, self.arms_ax, self.regret_ax = self.bandit_env.make_background(
                show_dists, show_regret
            )
        else:
            for line in self.plot_content:
                line.remove()

        self.plot_content = []

        for arm in range(self.num_arms):
            separation = 1.0 / self.num_arms
            location = -separation * (self.num_arms - arm)
            self.plot_content.extend(
                self.arms_ax.plot(
                    [location, location],
                    [np.mean(self.rewards[arm]), self.upper_confidence_bounds[arm]],
                    lw=2.5,
                    c=self.bandit_env.COLORS[arm],
                )
            )
            self.plot_content.extend(
                self.arms_ax.plot(
                    [
                        -separation * (self.num_arms - arm - 0.25),
                        -separation * (self.num_arms - arm + 0.25),
                    ],
                    [
                        self.upper_confidence_bounds[arm],
                        self.upper_confidence_bounds[arm],
                    ],
                    "-",
                    lw=3,
                    c=self.bandit_env.COLORS[arm],
                )
            )
            self.plot_content.extend(
                self.arms_ax.plot(
                    [location],
                    [np.mean(self.rewards[arm])],
                    ".",
                    ms=15,
                    c=self.bandit_env.COLORS[arm],
                )
            )

            if self.num_arms <= 8:
                self.plot_content.append(
                    self.arms_ax.text(
                        location,
                        -3,
                        r" Arm {}".format(arm),
                        ha="center",
                        va="center",
                        fontsize=min(60 / (self.num_arms), 10),
                    )
                )
                self.plot_content.append(
                    self.arms_ax.text(
                        location,
                        -4.6,
                        r"$n_{}={}$".format(arm, self.times_pulled[arm]),
                        ha="center",
                        va="center",
                        fontsize=min(
                            60 / (self.num_arms),
                            10,
                        ),
                    )
                )

        if show_dists:
            self.arms_ax.set_ylim([-10, 15])
        else:
            self.arms_ax.set_ylim([-13, 15])
        self.arms_ax.set_xticks([])
        self.arms_ax.set_ylabel("Rewards", fontsize=15)
        self.figure.suptitle("UCB Demo")
        return


if __name__ == "__main__":
    num_arms = 6
    T = 200
    num_runs = 1
    show = 1
    env = BanditEnv(num_arms, "rand")

    alg1 = UCB(env, 1.0 / (T) ** 2)
    ucb_regret = 0
    for run in range(num_runs):
        print("UCB Run: " + str(run))
        ucb_regret += np.array(alg1.run_Interactive(1, 0, 0))

    plt.plot(ucb_regret)
    plt.xlabel("Time (t)", fontsize=10)
    plt.ylabel("Pseudo-regret", fontsize=10)
    plt.show()
