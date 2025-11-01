import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

dirpath_base = Path(__file__).parent.resolve()

from neuron import h
h.nrn_load_dll(str(dirpath_base / 'nrnmech.dll'))
from netpyne import specs, sim

matplotlib.use('Qt5Agg', force=True)


# --------------------------
# Simulation / model config
# --------------------------
cfg = specs.SimConfig()
cfg.duration = 10000.0     # ms
cfg.dt = 0.05
cfg.hParams = {'celsius': 34.0}
cfg.verbose = False
cfg.recordStep = 0.1
cfg.spikeThreshold = -20

# Analysis
cfg.analysis = dict(
    plotRaster={'include': ['pop_e', 'pop_i'], 'timeRange': [0, cfg.duration]},
    plotTraces={'include': [('pop_e', 0), ('pop_i', 0)],
                'timeRange': [0, cfg.duration],
                'oneFigPer': 'cell'},
    plotRates={'timeRange': [0, cfg.duration], 'binSize': 10}
)

# Network sizes
NE = 20
NI = 20

# Background noise params
mu_0    = 1e-2
sigma_0 = 0.0
mu_e0 = mu_i0 = mu_0
sigma_e0 = sigma_i0 = sigma_0

# Rate control params
rE0  = 5.0   # Hz target E
rI0  = 8.0   # Hz target I
tau_ctrl = 200.0
k_ctrl = 5e-5
z0 = -0.35

# Reversal potentials
E_AMPA = 0.0    # mV
E_GABA = -70.0  # mV

# Synaptic properties
cee, cei, cie, cii = 0.5, 0.5, 0.5, 0.5
wee, wei, wie, wii = 0, 0, 0.0, 0

rng = np.random.default_rng(1234)  # reproducible

# --------------------------
# NetParams
# --------------------------
netParams = specs.NetParams()

# Synaptic mechanisms
netParams.synMechParams['AMPA'] = {'mod': 'ExpSyn', 'e': E_AMPA, 'tau': 2.0}
netParams.synMechParams['GABA'] = {'mod': 'ExpSyn', 'e': E_GABA, 'tau': 5.0}

# Cell rule: single-compartment with HH
netParams.cellParams['HH1C'] = {
    'secs': {
        'soma': {
            'geom': {'diam': 18.8, 'L': 18.8, 'Ra': 123.0},
            'mechs': {'hh': {'gnabar': 0.2, 'gkbar': 0.03, 'gl': 0.0002, 'el': -65}}
        }
    }
}

# Populations
netParams.popParams['pop_e'] = {'cellType': 'HH1C', 'numCells': NE}
netParams.popParams['pop_i'] = {'cellType': 'HH1C', 'numCells': NI}

# Connectivity helpers
def conn(rule_name, pre, post, prob, weight_uS, delay_ms=1.0, syn='ampa'):
    netParams.connParams[rule_name] = {
        'preConds':  {'pop': pre},
        'postConds': {'pop': post},
        'probability': prob,
        'weight': weight_uS,          # uS for ExpSyn
        'delay': delay_ms,
        'synMech': syn,
        'sec': 'soma',
        'loc': 0.5
    }

# E->E (AMPA), E->I (AMPA), I->E (GABA), I->I (GABA)
conn('E->E', 'pop_e', 'pop_e', cee, wee, syn='AMPA')
conn('E->I', 'pop_e', 'pop_i', cie, wie, syn='AMPA')
conn('I->E', 'pop_i', 'pop_e', cei, wei, syn='GABA')
conn('I->I', 'pop_i', 'pop_i', cii, wii, syn='GABA')

# --------------------------
# Build network
# --------------------------
sim.initialize(simConfig=cfg, netParams=netParams)
pc = sim.pc  # NetPyNE ParallelContext
rank = sim.rank
nhost = sim.nhosts

sim.net.createPops()
sim.net.createCells()
sim.net.connectCells()
sim.net.addStims()

sim.setupRecording()

# NetPyNE already registers gids & sources for synapses/spike recording.
# We'll only *subscribe* to those gids with gid_connect per-rank controllers.

# Helper: local cells by pop
def get_local_cells(pop_name):
    return [c for c in sim.net.cells if c.tags['pop'] == pop_name]

# Helper: all gids by pop (global)
def get_all_gids(pop_name):
    return sim.net.pops[pop_name].cellGids[:]  # full list of gids

# --------------------------
# Controllers (per-pop per-rank)
# --------------------------
controllers = {}          # pop -> controller (or None)
controller_debug = {}     # pop -> dict of debug vectors/lists (only local)
pop_sizes = {'pop_e': NE, 'pop_i': NI}

def make_controller(pop_name, r0_target):
    local_cells = get_local_cells(pop_name)
    if len(local_cells) == 0:
        controllers[pop_name] = None
        controller_debug[pop_name] = {}
        return

    # Attach to *any* local soma (only to expose POINTERs for local cells)
    soma = local_cells[0].secs['soma']['hObj']
    ctrl = h.PopController(soma(0.5))
    ctrl.tau, ctrl.r0, ctrl.k, ctrl.z0 = tau_ctrl, r0_target, k_ctrl, z0
    controllers[pop_name] = ctrl

    # Subscribe this controller to spikes from *all gids* of this population.
    # This creates incoming NetCons on THIS rank that deliver events from each gid.
    debug = {'nc_in_list': [], 'ev_list': []}
    for gid in get_all_gids(pop_name):
        nc_in = pc.gid_connect(gid, ctrl)  # auto-routed across ranks
        nc_in.weight[0] = 1.0 / pop_sizes[pop_name]
        nc_in.delay = 0.0
        debug['nc_in_list'].append(nc_in)
        #debug['ev_list'].append(ev)
    controller_debug[pop_name] = debug

make_controller('pop_e', rE0)
make_controller('pop_i', rI0)

# --------------------------
# Background noise clamps (local cells only)
# --------------------------
def make_noise_vectors(ncell, dur_ms, dt_ms, seed0, zero_mean=True):
    nstep = int(math.ceil(dur_ms/dt_ms)) + 1
    rng_local = np.random.default_rng(seed0)
    noises = []
    for _ in range(ncell):
        x = rng_local.standard_normal(nstep).astype(np.float64)
        if zero_mean:
            x -= x.mean()
        t = np.arange(nstep, dtype=np.float64) * dt_ms
        noises.append((t, x))
    return noises

# Generate per-pop noise only for the number of *local* cells
cells_e_local = get_local_cells('pop_e')
cells_i_local = get_local_cells('pop_i')
noises_e = make_noise_vectors(len(cells_e_local), cfg.duration, cfg.dt, seed0=(42 + rank))
noises_i = make_noise_vectors(len(cells_i_local), cfg.duration, cfg.dt, seed0=(4242 + rank))

def attach_noise_local(cells_local, noises, controller, label, mu0=None, sigma0=None):
    for (c, (t, x)) in zip(cells_local, noises):
        soma = c.secs['soma']['hObj']
        clamp = h.BgNoiseClamp(soma(0.5))
        if mu0 is not None:
            clamp.mu = mu0
        if sigma0 is not None:
            clamp.sigma = sigma0
        # POINTER only to *local* controller on THIS rank
        if controller is not None:
            # h.setpointer(controller._ref_z, 'pmu', clamp)
            h.setpointer(controller._ref_z, 'psigma', clamp)
        # play the zero-mean noise
        tvec = h.Vector(t); xvec = h.Vector(x)
        xvec.play(clamp._ref_noise, tvec, 1)  # piecewise constant per dt
        # keep refs to avoid GC
        c.secs['soma'].setdefault('stims', []).append(
            {'type': 'BgNoiseClamp', 'hObj': clamp, 'tvec': tvec, 'xvec': xvec, 'label': label}
        )

attach_noise_local(cells_e_local, noises_e, controllers['pop_e'], 'Ebg', mu0=mu_e0, sigma0=sigma_e0)
attach_noise_local(cells_i_local, noises_i, controllers['pop_i'], 'Ibg', mu0=mu_i0, sigma0=sigma_i0)

# --------------------------
# Recording (rank-aware)
# --------------------------
# Only rank 0 will record/plot controller traces for simplicity.
tvec = h.Vector()
zE_vec = h.Vector(); zI_vec = h.Vector()
rE_vec = h.Vector(); rI_vec = h.Vector()
vE_vec = h.Vector(); vI_vec = h.Vector()
mE_vec = h.Vector(); hE_vec = h.Vector(); nE_vec = h.Vector()

if rank == 0:
    tvec.record(h._ref_t)
    if controllers['pop_e'] is not None:
        zE_vec.record(controllers['pop_e']._ref_z)
        rE_vec.record(controllers['pop_e']._ref_rate)
    if controllers['pop_i'] is not None:
        zI_vec.record(controllers['pop_i']._ref_z)
        rI_vec.record(controllers['pop_i']._ref_rate)
    # pick *any* local cell on rank 0 for V and gates (if present)
    if len(cells_e_local) > 0:
        segE = cells_e_local[0].secs['soma']['hObj'](0.5)
        vE_vec.record(segE._ref_v)
        mE_vec.record(segE.hh._ref_m)
        hE_vec.record(segE.hh._ref_h)
        nE_vec.record(segE.hh._ref_n)
    if len(cells_i_local) > 0:
        segI = cells_i_local[0].secs['soma']['hObj'](0.5)
        vI_vec.record(segI._ref_v)

# --------------------------
# Run, gather
# --------------------------
sim.runSim()
sim.gatherData()  # gathers spikes etc. to rank 0

# --------------------------
# Rank 0: print, store, plot
# --------------------------
if rank == 0:
    # Save controller traces into sim data for plotting
    sim.allSimData['t_ctrl']  = np.array(tvec) if len(tvec) else np.array([])
    sim.allSimData['zE']      = np.array(zE_vec) if len(zE_vec) else np.array([])
    sim.allSimData['zI']      = np.array(zI_vec) if len(zI_vec) else np.array([])
    sim.allSimData['rateE']   = np.array(rE_vec) if len(rE_vec) else np.array([])
    sim.allSimData['rateI']   = np.array(rI_vec) if len(rI_vec) else np.array([])
    sim.allSimData['vE']      = np.array(vE_vec) if len(vE_vec) else np.array([])
    sim.allSimData['vI']      = np.array(vI_vec) if len(vI_vec) else np.array([])

    # Basic printed summary
    def pop_mean_rate(sim, pop_name: str, duration_ms: float) -> float:
        spkt  = sim.allSimData.get('spkt', [])
        spkid = sim.allSimData.get('spkid', [])
        if spkt is None or spkid is None or len(spkt) == 0:
            return 0.0
        gids = sim.net.pops[pop_name].cellGids
        if not gids:
            return 0.0
        mask = np.isin(spkid, gids)
        n_spikes = int(np.sum(mask))
        dur_s = float(duration_ms) / 1000.0
        return n_spikes / (len(gids) * dur_s)

    rE = pop_mean_rate(sim, 'pop_e', cfg.duration)
    rI = pop_mean_rate(sim, 'pop_i', cfg.duration)
    print(f'\nMean rates: E={rE:.2f} Hz (target {rE0}), I={rI:.2f} Hz (target {rI0})')

    if len(zE_vec):
        print(f'Final zE={sim.allSimData["zE"][-1]:.3f}')
    if len(zI_vec):
        print(f'Final zI={sim.allSimData["zI"][-1]:.3f}')

    # Controller event counts (first few) if available
    if controller_debug['pop_e'].get('ev_list'):
        counts = []
        for ev in controller_debug['pop_e']['ev_list'][:min(5, len(controller_debug['pop_e']['ev_list']))]:
            counts.append(int(ev.size()) if ev is not None else -1)
        print('Num. controller events (E, sample):', counts)

    # --------------------------
    # Custom plots (rank 0)
    # --------------------------
    if len(sim.allSimData['t_ctrl']) > 0:
        plt.figure(figsize=(8, 5))

        plt.subplot(3, 1, 1)
        if len(sim.allSimData['zE']): plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['zE'], label='z_E')
        if len(sim.allSimData['zI']): plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['zI'], label='z_I')
        plt.title('Rate controller output'); plt.legend(); plt.xlim(0, cfg.duration)

        plt.subplot(3, 1, 2)
        if len(sim.allSimData['vE']): plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['vE'], label='v_E')
        if len(sim.allSimData['vI']): plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['vI'], label='v_I')
        plt.title('Membrane voltage'); plt.legend(); plt.xlim(0, cfg.duration)

        plt.subplot(3, 1, 3)
        if len(sim.allSimData['rateE']): plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['rateE'], label='rate_E')
        if len(sim.allSimData['rateI']): plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['rateI'], label='rate_I')
        plt.axhline(rE0, ls='--', alpha=0.5)
        plt.axhline(rI0, ls='--', alpha=0.5)
        plt.title('Firing rate (Hz)'); plt.xlabel('Time (ms)'); plt.legend()
        plt.xlim(0, cfg.duration); plt.ylim(0, 25)

        plt.tight_layout()
        plt.show()

# (Optional) built-in plots are safe to call on rank 0 after gather
# if rank == 0:
#     sim.analysis.plotRaster(include=['pop_e','pop_i'])
#     sim.analysis.plotTraces(include=[('pop_e',0),('pop_i',0)], oneFigPer='cell')
#     sim.analysis.plotRates()
