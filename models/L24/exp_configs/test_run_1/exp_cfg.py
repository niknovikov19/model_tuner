
def apply_exp_cfg(cfg):
    
    cfg.duration = 1 * 1e3
    
    cfg.connected = True
    cfg.equal_inputs = False
    
    pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
    
    cfg.analysis['plotRaster'] = {
        'include': pop_names,
        'saveFig': True, 'showFig': False,
        'popRates': True, 'orderInverse': True,
        'timeRange': [0, cfg.duration],
        'figSize': (14,12), 'lw': 0.3,
        'markerSize': 3, 'marker': '.', 'dpi': 300
    }
    cfg.analysis['plotSpikeStats'] = {
        'include': pop_names,
        'stats': ['rate', 'isicv'],
        'showFig': False, 'saveFig': True,
        'timeRange': [500, cfg.duration],
        'figSize': (6,12), 'dpi': 300
    }
