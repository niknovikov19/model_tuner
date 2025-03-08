from netpyne.batchtools import specs


def create_base_cfg():

    cfg = specs.SimConfig()

    cfg.seeds['stim'] = 3
    cfg.duration = 500
    cfg.dt = 0.05
    cfg.verbose = False
    cfg.seeds['m'] = 123
    cfg.simLabel = "SIM1"
    
    cfg.connected = True
    
    cfg.equal_inputs = False
    cfg.ext_inp_rate_common: 0  # used if equal_inputs==True
    
    cfg.wmult = 1
    
    # Input rates (used if equal_inputs==False)
    cfg.ext_input_rates = {
        'L2e': 65,
        'L2i': 110,
        'L4e': 115,
        'L4i': 280,
    }
    
    pop_names = ['L2e', 'L2i', 'L4e', 'L4i']
     
    cfg.recordCellsSpikes = pop_names
    
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
    
    return cfg