from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.optimize import curve_fit


class ModelDesc:
    def __init__(self): pass
    def get_pop_names(self) -> List[str]: pass

    
@dataclass
class PopRegime:
    pass

@dataclass
class NetRegime:
    pop_regimes: Dict[str, PopRegime] = field(default_factory=dict)
    
    def __init__(self): pass  # constructor should be defined in children
    
    def get_pop_names(self):
        return list(self.pop_regimes.keys())
    
    def get_pop_attr_vec(self, attr):
        return np.array([getattr(R, attr) for R in self.pop_regimes.values()])
    
@dataclass
class NetRegimeList:
    net_regimes: List[NetRegime] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
    
    def check_pop_consistency(self) -> bool:
        for R in self.net_regimes:
            if R.get_pop_names() != self.net_regimes[0].get_pop_names():
                return False
        return True
    
    def get_pop_attr_mat(self, attr):
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
        M = [R.get_pop_attr_vec(attr).reshape(-1, 1) for R in self.net_regimes]
        return np.concatenate(M, axis=1)        


@dataclass      
class PopInput:
    pass

@dataclass
class NetInput:
    pop_inputs: Dict[str, PopInput] = {}

    
class PopIRMapper:
    def I_to_R(self, I: PopInput) -> PopRegime: pass
    def R_to_I(self, R: PopRegime) -> PopInput: pass
    
class NetIRMapper:
    def __init__(self):
        self.pop_IR_mappers: Dict[str, PopIRMapper] = {}
        
    def set_pop_mapper(self, pop_name: str, mapper: PopIRMapper):
        self.pop_IR_mappers[pop_name] = mapper
        
    def _I_to_R(self, I: NetInput) -> NetRegime():
        R = NetRegime()
        for name, I_ in I.pop_inputs.items():
            R.pop_regimes[name] = self.pop_IR_mappers[name].I_to_R(I_)
        return R
    
    def R_to_I(self, R: NetRegime) -> NetInput():
        I = NetInput()
        for name, R_ in R.pop_regimes.items():
            I.pop_inputs[name] = self.pop_IR_mappers[name].R_to_I(R_)
        return I


class NetUCMapper:
    def Ru_to_Rc(self, Ru: NetRegime) -> NetRegime: pass  # unconnected -> connected
    def Rc_to_Ru(self, Rc: NetRegime) -> NetRegime: pass  # connected -> unconnected


@dataclass
class PopParamsWC:
    gain: float = 1
    thresh: float = 1
    
class ModelDescWC(ModelDesc):    
    def __init__(self, num_pops):
        self.super().init()
        self.pops = {}
        for n in range(num_pops):
            self.pops[f'pop{n}'] = ModelDescWC.PopParamsWC()
        self.conn = 2 * np.random.rand(num_pops, num_pops) - 1
    
    def get_pop_names(self) -> List[str]:
        return list(self.pops.keys())

@dataclass        
class PopRegimeWC(PopRegime):
    r: float = 0
    
@dataclass
class PopInputWC(PopInput):
    I: float = 0

@dataclass
class NetRegimeWC(NetRegime):
    def __init__(self, pop_names: List[str] = None, pop_rates: List[float] = None):
        self.pop_regimes = {}
        if pop_names is not None:
            for pop_name, r in zip(pop_names, pop_rates):
                self.pop_regimes[pop_name] = PopRegimeWC(r=r)
    
    def get_pop_rates(self) -> np.ndarray:
        return np.array([R_.r for R_ in self.pop_regimes.values()])

class PopIRMapperWC(PopIRMapper):
    def __init__(self, pop_params: PopParamsWC):
        self.pop_params = pop_params
        
    def I_to_R(self, I: PopInputWC) -> PopRegimeWC:
        g, th = self.pop_params.gain, self.pop_params.thresh
        r = 1 / (1 + np.exp(-g * (I.I - th)))
        return PopRegimeWC(r=r)

    def R_to_I(self, R: PopRegime) -> PopInput:
        g, th = self.pop_params.gain, self.pop_params.thresh
        I = -np.log(1 / R.r - 1) / g + th
        return PopInputWC(I=I)

class NetIRMapperWC(NetIRMapper):
    def __init__(self, model: ModelDescWC):
        super().init(self)
        for name, pop in model.pops:
            self.set_pop_mapper(name, PopIRMapperWC(pop))

class NetUCMapperWC(NetUCMapper):
    def __init__(self):
        self.pop_names = []
        self.is_identity = True
        self.a, self.b, self.k = [], [], []
    
    def set_to_identity(self):
        self.is_identity = True
    
    def Ru_to_Rc(self, Ru: NetRegimeWC) -> NetRegimeWC:
        if Ru.get_pop_names() != self.pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        rr_u = Ru.get_pop_rates()
        if self.is_identity:
            rr_c = rr_u
        else:
            rr_c = self.a * np.exp(self.b * rr_u) + self.k
        return NetRegimeWC(self.pop_names, rr_c)
        
    def Rc_to_Ru(self, Rc: NetRegimeWC) -> NetRegimeWC:
        if Rc.get_pop_names() != self.pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        rr_c = Rc.get_pop_rates()
        if self.is_identity:
            rr_u = rr_c
        else:
            rr_u = np.log((rr_c - self.k) / self.a) / self.b
        return NetRegimeWC(self.pop_names, rr_u)
    
    def fit_from_data(self, Ru_Rc_pairs: List[Tuple[NetRegimeWC, NetRegimeWC]]):
        self.is_identity = False
        self.pop_names = Ru_Rc_pairs[0][0].get_pop_names()
        for Ru, Rc in Ru_Rc_pairs:
            if ((Ru.get_pop_names() != self.pop_names) or
                (Rc.get_pop_names() != self.pop_names)):
                raise ValueError('All Ru and Rc entries should have the same pops.')                
        npops = len(self.pop_names)
        self.a, self.b, self.k = (np.zeros(npops) for _ in range(3))
        def fit_func(rr_u, a, b, k):
            return a * np.exp(b * rr_u) + k
        for n, pop in enumerate(pop_names):
            rr_u = np.array([RuRc[0].pop_regimes[pop].r for RuRc in Ru_Rc_pairs])
            rr_c = np.array([RuRc[1].pop_regimes[pop].r for RuRc in Ru_Rc_pairs])
            par0 = [1.0, 1.0, 1.0]
            par, _ = curve_fit(model, rr_u, rr_c, p0=par0)
            self.a[n], self.b[n], self.k[n] = par
            

# Network of Wilson-Cowan populations
npops = 5
model = ModelDescWC(num_pops=npops)
pop_names = model.get_pop_names()

# Original target regime (vector of pop. firing rates)
rr_base = np.arange(npops) + 1

# P_FR
pfr_vec = np.linspace(0, 1, 10)

# Target regimes
R0_vec = [NetRegimeWC(pop_names, rr_base * pfr)
          for pfr in pfr_vec]

# I-R mapper, explicitly uses Wilson-Cowan gain functions of populations
ir_mapper = NetIRMapperWC(model)

# Unconnected-to-connected regime mapper


n = 0
R0 = R0_vec[n]

R_iter = [R0]

for iter_num in range(10):
    R = R_iter[-1]
    I = ir_mapper.R_to_I(R)
    P = I.apply_to_net_params(P_base)
    req = SimRequest(cfg, P)
    req_name = f'req_{n}_{iter_num}'
    disp.submit_sim_request(req_name, req)
    sim_res = disp.get_sim_result(req_name)
    R_next = NetRegime.calc_from_sim_result(sim_res)
    R_iter.append(R_next)




# =============================================================================
# class NetRegime:
#     def __init__(self, fpath): pass
# 
#     @classmethod
#     def calc_from_sim_result(cls, sim_res: SimResult) -> 'NetRegime':
#         pass
# 
# def net_regime_from_pfr(R: NetRegime, pfr: float) -> NetRegime:
#     pass
# 
# class NetInput:
#     def __init__(self): pass
#     def apply_to_net_params(par: NetParams) -> NetParams: pass
# =============================================================================


class NetParams:
    def __init__(self): pass

class SimConfig:
    def __init__(self): pass

class ModelDescNP(ModelDesc):
    def __init__(self):
        self.net_params = NetParams()
        self.sim_cfg = SimConfig()


R_base = NetRegime('R_base.json')

pfr_vec = np.linspace(0, 1, 10)
R0_vec = [net_regime_from_pfr(R_base, pfr) for pfr in pfr_vec]

ir_mapper = IRMapper()

P_base = NetParams()
cfg = SimConfig()

disp = SimRequestDispatcher()

n = 0
R0 = R0_vec[n]

R_iter = [R0]

for iter_num in range(10):
    R = R_iter[-1]
    I = ir_mapper.R_to_I(R)
    P = I.apply_to_net_params(P_base)
    req = SimRequest(cfg, P)
    req_name = f'req_{n}_{iter_num}'
    disp.submit_sim_request(req_name, req)
    sim_res = disp.get_sim_result(req_name)
    R_next = NetRegime.calc_from_sim_result(sim_res)
    R_iter.append(R_next)


class SimRequest:
    def __init__(self, cfg: SimConfig, net_params: NetParams):
        self.cfg = cfg
        self.net_params = net_params

class SimResult:
    def __init__(self): pass
        
class SimRequestDispatcher:
    def submit_sim_request(self, name, req: SimRequest): pass
    def get_sim_result(self, name) -> SimResult: pass