import matplotlib.pyplot as plt
import pandas as pd


def plot_events(
    power_df: pd.DataFrame, events_df: pd.DataFrame, states_df: pd.DataFrame, ax=None
):
    if ax is None:
        ax = plt.gca()
    ax.plot(power_df.index, power_df.values, label="Signal", color="tab:blue")
    ax.scatter(
        events_df.index, power_df.loc[events_df.index], label="Event", color="tab:red"
    )
    ax.step(
        states_df.index,
        states_df.values,
        where="post",
        linestyle="dashed",
        label="State",
        color="tab:green",
    )
    ax.set_ylabel("Power [kW]")
    ax.set_xlabel("Date")


def plot_statistics(statistics: pd.DataFrame, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.stem(statistics.index, statistics, label="Event indicator")
    ax.set_ylabel("Statistic")
    ax.set_xlabel("Date")
    ax.legend()
