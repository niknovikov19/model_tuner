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
    
    # Input rates (used if equal_inputs==False)
    cfg.ext_input_rates = {
        'L2e': 65,
        'L2i': 110,
        'L4e': 115,
        'L4i': 280,
    }
     
    cfg.recordCellsSpikes = ['L2e', 'L2i', 'L4e', 'L4i']
    
    return cfg