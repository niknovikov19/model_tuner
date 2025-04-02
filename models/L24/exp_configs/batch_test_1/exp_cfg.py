
def apply_exp_cfg(cfg):
    
    cfg.duration = 2 * 1e3
    
    # Unconnected network with the same external input rate for every pop.
    cfg.connected = False
    cfg.equal_inputs = True
    cfg.ext_inp_rate_common = 0  # will be set by batchtools
    
    pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
    
    cfg.analysis['plotRaster'] = {
        'include': pop_names, 'saveFig': True, 'showFig': False,
        'popRates': True, 'orderInverse': True, 'timeRange': [0, cfg.duration],
        'figSize': (14, 12), 'lw': 0.3, 'markerSize': 3, 'marker': '.', 'dpi': 300
    }
    cfg.analysis['plotSpikeStats'] = {
        'stats': ['rate', 'isicv'], 'figSize': (6, 12),
        'timeRange': [1000, cfg.duration], 'dpi': 300, 'showFig': 0, 'saveFig': 1
    }
