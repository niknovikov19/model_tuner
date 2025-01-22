from typing import Dict, List, Tuple

import numpy as np


def calc_pop_rate(
        pop_spikes: List[np.ndarray],  # per cells or combined into 1 entry
        time_limits: Tuple[float],
        ncells: int = 1
        ) -> List[float]:
    rates = []
    T = time_limits[1] - time_limits[0]
    for spike_times in pop_spikes:
        nspikes = np.sum((spike_times >= time_limits[0]) &
                         (spike_times <= time_limits[1]))
        rates.append(nspikes / T / ncells)
    return rates

def calc_net_rates(
        net_spikes: Dict[str, List[np.ndarray]],  # {pop: spikes}
        time_limits: Tuple[float],
        ncells: Dict[str, int] | None = None,
        pop_names: List[str] | None = None
        ) -> Dict[str, List[float]]:  # {pop: rates}
    net_rates = {}
    pop_names = pop_names or list(net_spikes)
    ncells = ncells or {pop_name: 1 for pop_name in pop_names}
    for pop_name in pop_names:
        net_rates[pop_name] = calc_pop_rate(
            net_spikes[pop_name], time_limits, ncells[pop_name]
        )
    return net_rates


# =============================================================================
# def calc_rate_dynamics(spike_times, time_range, dt, pop_sz=1,
#                        epoch_len=None):
#     """Calculate firing rate dynamics from combined spiketrains. """
#     t1 = time_range[0]
#     t2 = time_range[1]
#     # Decrease the time range so it is a multiple of the epoch
#     if epoch_len is not None:
#         num_epochs = np.floor((time_range[1] - time_range[0]) / epoch_len)
#         t2 = t1 + epoch_len * num_epochs
#     else:
#         num_epochs = 1
#     # Get spike times within the given time range
#     spike_times = np.array(spike_times)
#     mask = (spike_times >= t1) & (spike_times <= t2)
#     spike_times = spike_times[mask]
#     # Put all the spikes into a single epoch
#     if epoch_len is not None:
#         spike_times = ((spike_times - t1) % epoch_len) + t1
#         t2 = t1 + epoch_len
#     # Transform: spike time -> sample number
#     Nbins = int((t2 - t1) / dt)
#     #spike_times = np.sort(spike_times)
#     bin_idx = np.floor((spike_times - t1) / dt)
#     bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < Nbins)]
#     bin_idx = bin_idx.astype(np.int64)
#     # Calculate firing rate dynamics
#     rvec = np.bincount(bin_idx, minlength=Nbins)
#     rvec = rvec / (dt * pop_sz * num_epochs)
#     # Time samples
#     tvec = np.arange(Nbins, dtype=np.float64) * dt + t1
#     # Return the result
#     return tvec, rvec
# =============================================================================
