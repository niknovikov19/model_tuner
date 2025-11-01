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
mu_0    = -5e-2
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
wee, wei, wie, wii = 0.005, 0.1, 0.005, 0.1

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

""" netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 10, 'noise': 0.5}
netParams.stimTargetParams['bkg->PYR'] = {
    'source': 'bkg',
    'conds': {'pop': 'pop_e'},
    'weight': 0.1,
    'delay': 5,
    'synMech': 'AMPA'
} """

# --------------------------
# Build network (stepwise so we can attach custom mechanisms)
# --------------------------
sim.initialize(simConfig=cfg, netParams=netParams)
sim.net.createPops()
sim.net.createCells()
sim.net.connectCells()
sim.net.addStims()

sim.setupRecording()

# Create one PopController per population; attach to any soma in that pop
cells_e = [c for c in sim.net.cells if c.tags['pop'] == 'pop_e']
cells_i = [c for c in sim.net.cells if c.tags['pop'] == 'pop_i']

assert len(cells_e) == NE and len(cells_i) == NI

# Attach controllers to the first cell in each pop
se_e = cells_e[0].secs['soma']['hObj']
se_i = cells_i[0].secs['soma']['hObj']
ctrlE = h.PopController(se_e(0.5))
ctrlI = h.PopController(se_i(0.5))
ctrlE.tau, ctrlE.r0, ctrlE.k, ctrlE.z0 = tau_ctrl, rE0, k_ctrl, z0
ctrlI.tau, ctrlI.r0, ctrlI.k, ctrlI.z0 = tau_ctrl, rI0, k_ctrl, z0

# Connect NetCons from each cell to its population controller
def attach_spike_feed(cells, controller, pop_size):
    for c in cells:
        soma = c.secs['soma']['hObj']
        apc = h.APCount(soma(0.5))      # robust spike detector
        apc.thresh = cfg.spikeThreshold
        #nc = h.NetCon(apc, controller)  # send events to controller
        #nc = h.NetCon(apc, None)
        nc = h.NetCon(soma(0.5)._ref_v, controller, sec=soma)
        #nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)

        #dummy = h.ExpSyn(soma(0.5))
        #nc = h.NetCon(apc, dummy)

        nc.weight[0] = 1.0 / pop_size     # scale to “per-cell” contribution
        #nc.weight[0] = 0
        nc.threshold = cfg.spikeThreshold
        nc.delay = 0.0

        if 'debug' not in c.secs['soma']:
            c.secs['soma']['debug'] = {}
        # keep Python refs so nothing gets GC'd
        dbg = c.secs['soma'].setdefault('debug', {})
        dbg.setdefault('apc_list', []).append(apc)
        dbg.setdefault('nc_list', []).append(nc)
        #dbg.setdefault('dummy_list', []).append(dummy)

        # record event times delivered to the controller
        ev = h.Vector()
        #h.CVode().record(nc, ev)
        #nc.record(ev)
        apc.record(ev)
        dbg.setdefault('ev_list', []).append(ev)

attach_spike_feed(cells_e, ctrlE, pop_size=len(cells_e))
attach_spike_feed(cells_i, ctrlI, pop_size=len(cells_i))

# --------------------------
# Background noise clamps
# --------------------------
def make_noise_vectors(ncell, dur_ms, dt_ms, seed0, zero_mean=True):
    nstep = int(math.ceil(dur_ms/dt_ms)) + 1
    rng_local = np.random.default_rng(seed0)
    noises = []
    for _ in range(ncell):
        # Gaussian white noise, zero-mean unit variance; you can replace with OU if desired
        x = rng_local.standard_normal(nstep).astype(np.float64)
        if zero_mean:
            x -= x.mean()
        t = np.arange(nstep, dtype=np.float64) * dt_ms
        noises.append((t, x))
    return noises

noises_e = make_noise_vectors(NE, cfg.duration, cfg.dt, seed0=42)
noises_i = make_noise_vectors(NI, cfg.duration, cfg.dt, seed0=4242)

# Attach BgNoiseClamp to every soma; play zero-mean noise into .noise; set POINTER pmu to controller.mu; set sigma
def attach_noise(cells, noises, controller, label, mu0=None, sigma0=None):
    for c, (t, x) in zip(cells, noises):
        soma = c.secs['soma']['hObj']
        clamp = h.BgNoiseClamp(soma(0.5))
        if mu0 is not None:
            clamp.mu = mu0
        if sigma0 is not None:
            clamp.sigma = sigma0
        # pointer to shared mu/sigma of this population
        if controller is not None:
            #h.setpointer(controller._ref_z, 'pmu', clamp)
            h.setpointer(controller._ref_z, 'psigma', clamp)
        # play the zero-mean noise into RANGE noise
        tvec = h.Vector(t)
        xvec = h.Vector(x)
        xvec.play(clamp._ref_noise, tvec, 1)  # interpolate=1: piecewise constant per dt
        # keep references to avoid GC
        if 'stims' not in c.secs['soma']:
            c.secs['soma']['stims'] = []
        c.secs['soma']['stims'].append({'type': 'BgNoiseClamp', 'hObj': clamp,
                                        'tvec': tvec, 'xvec': xvec, 'label': label})

attach_noise(cells_e, noises_e, ctrlE, 'Ebg', mu0=mu_e0, sigma0=sigma_e0)
attach_noise(cells_i, noises_i, ctrlI, 'Ibg', mu0=mu_i0, sigma0=sigma_i0)

# --------------------------
# Record controller signals for analysis
# --------------------------
# We’ll sample controller mu and rate with a Vector record
tvec = h.Vector().record(h._ref_t)
zE_vec  = h.Vector().record(ctrlE._ref_z)
zI_vec  = h.Vector().record(ctrlI._ref_z)
rE_vec   = h.Vector().record(ctrlE._ref_rate)
rI_vec   = h.Vector().record(ctrlI._ref_rate)
""" iE = h.Vector().record(
    cells_e[0].secs['soma']['stims'][0]['hObj']._ref_i)
iI = h.Vector().record(
    cells_i[0].secs['soma']['stims'][0]['hObj']._ref_i) """
vE_vec = h.Vector().record(cells_e[0].secs['soma']['hObj'](0.5)._ref_v)
vI_vec = h.Vector().record(cells_i[0].secs['soma']['hObj'](0.5)._ref_v)


# Record HH gates (E cell)
segE = cells_e[0].secs['soma']['hObj'](0.5)
mE_vec = h.Vector().record(segE.hh._ref_m)
hE_vec = h.Vector().record(segE.hh._ref_h)
nE_vec = h.Vector().record(segE.hh._ref_n)

""" bkg_amp = 0.1
icl = h.IClamp(cells_e[0].secs['soma']['hObj'](0.5))
icl.delay = 100
icl.dur = 1000
icl.amp = bkg_amp """

# --------------------------
# Run, gather
# --------------------------

#h.finitialize(-70)
#h.fcurrent() 

#sim.setupRecording()

sim.runSim()
sim.gatherData()

def count_upcrossings(v, t, thr):
    # simple upward-crossing counter (linear interpolation)
    import numpy as np
    v = np.asarray(v); t = np.asarray(t)
    below = v[:-1] < thr
    above = v[1:]  >= thr
    return int(np.count_nonzero(below & above))

""" for i in range(min(3, len(cells_e))):
    d = cells_e[i].secs['soma']['debug']['probe']
    n_ev = int(d['ev'].size())                           # NetCon-recorded events
    n_x  = count_upcrossings(np.array(d['v']), np.array(d['t']), d['thresh'])
    vmin, vmax = float(min(d['v'])), float(max(d['v']))
    print(f"cell {i}: NetCon ev={n_ev}, upcross={n_x}, Vmin..Vmax=({vmin:.1f},{vmax:.1f}), thr={d['thresh']}") """

# Save controller traces into sim data for plotting
sim.allSimData['t_ctrl']  = np.array(tvec)
sim.allSimData['zE']     = np.array(zE_vec)
sim.allSimData['zI']     = np.array(zI_vec)
sim.allSimData['rateE']   = np.array(rE_vec)
sim.allSimData['rateI']   = np.array(rI_vec)
#sim.allSimData['iE']      = np.array(iE)
#sim.allSimData['iI']      = np.array(iI)
sim.allSimData['vE']      = np.array(vE_vec)
sim.allSimData['vI']      = np.array(vI_vec)

d = cells_e[0].secs['soma']['debug']
print("APCount n =", int(d['apc_list'][0].n))
print("events =", int(d['ev_list'][0].size()))
print("Vmax E0 =", float(np.max(sim.allSimData['vE'])))

print('Controller rate max.: ', np.max(sim.allSimData['rateE']))

# Basic printed summary
def pop_mean_rate(sim, pop_name: str, duration_ms: float) -> float:
    # pull spike arrays safely
    spkt  = sim.allSimData.get('spkt', [])
    spkid = sim.allSimData.get('spkid', [])
    if spkt is None or spkid is None or len(spkt) == 0:
        return 0.0

    gids = sim.net.pops[pop_name].cellGids
    if not gids:
        return 0.0

    # count spikes from this pop
    mask = np.isin(spkid, gids)
    n_spikes = int(np.sum(mask))

    dur_s = float(duration_ms) / 1000.0
    return n_spikes / (len(gids) * dur_s)

rE = pop_mean_rate(sim, 'pop_e', cfg.duration)
rI = pop_mean_rate(sim, 'pop_i', cfg.duration)
print(f'\nMean rates: E={rE:.2f} Hz (target {rE0}), I={rI:.2f} Hz (target {rI0})')
print(f'Final zE={sim.allSimData["zE"][-1]:.3f} nA, zI={sim.allSimData["zI"][-1]:.3f} nA')

# How many events hit the controller from first three E-cells?
print('Num. events: ', [
    int(cells_e[i].secs['soma']['debug']['ev_list'][0].size())
    #-1
    for i in range(min(1e6, len(cells_e)))
])

mE, hE, nE = np.array(mE_vec), np.array(hE_vec), np.array(nE_vec)

# --------------------------
# Custom plots: controller traces
# --------------------------
#plt.ion()
plt.figure(figsize=(8, 5))

plt.subplot(3, 1, 1)
plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['zE'], label='z_E')
plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['zI'], label='z_I')
plt.title('Rate controller output')
#plt.plot(sim.allSimData['t_ctrl'], mE, label='m')
#plt.plot(sim.allSimData['t_ctrl'], hE, label='h')
#plt.plot(sim.allSimData['t_ctrl'], nE, label='n')
plt.legend()
plt.xlim(0, cfg.duration)

plt.subplot(3, 1, 2)
plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['vE'], label='v_E')
plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['vI'], label='v_I')
#plt.title('Delivered background current')
plt.title('Membrane voltage')
plt.legend()
plt.xlim(0, cfg.duration)

plt.subplot(3, 1, 3)
plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['rateE'], label='rate_E (Hz)')
plt.plot(sim.allSimData['t_ctrl'], sim.allSimData['rateI'], label='rate_I (Hz)')
plt.axhline(rE0, ls='--', alpha=0.5)
plt.axhline(rI0, ls='--', alpha=0.5)
plt.title('Firing rate (Hz)')
plt.xlabel('Time (ms)')
plt.legend()
plt.xlim(0, cfg.duration)
plt.ylim(0, 25)

plt.show()

# Built-in NetPyNE analysis plots (raster, traces, rates)
#sim.analysis.plotRaster(include=['pop_e','pop_i'])
#input('Press any key')
#sim.analysis.plotTraces(include=[('pop_e',0),('pop_i',0)], oneFigPer='cell')
#sim.analysis.plotRates()
