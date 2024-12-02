from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
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
    
    def get_pop_attr_vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(R, attr) for R in self.pop_regimes.values()])
    
@dataclass
class NetRegimeList:
    net_regimes: List[NetRegime] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
    
    def __getitem__(self, n: int) -> NetRegime:
        return self.net_regimes[n]
    
    def __setitem__(self, n: int, R: NetRegime):
        self.net_regimes[n] = R
    
    def __len__(self) -> int:
        return len(self.net_regimes)
    
    def copy(self) -> 'NetRegimeList':
        return deepcopy(self)
    
    def check_pop_consistency(self) -> bool:
        for R in self.net_regimes:
            if R.get_pop_names() != self.net_regimes[0].get_pop_names():
                return False
        return True
    
    def get_pop_names(self):
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
        return self.net_regimes[0].get_pop_names()
    
    def get_pop_attr_mat(self, attr: str) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        if not self.check_pop_consistency():
            raise ValueError('All entries of NetRegimeList should have the same pops.')
        M = [R.get_pop_attr_vec(attr).reshape(-1, 1) for R in self.net_regimes]
        return np.concatenate(M, axis=1)        


@dataclass      
class PopInput:
    pass

@dataclass
class NetInput:
    pop_inputs: Dict[str, PopInput] = field(default_factory=dict)
    
    def get_pop_names(self):
        return list(self.pop_regimes.keys())
    
    def get_pop_attr_vec(self, attr: str) -> np.ndarray:
        return np.array([getattr(I, attr) for I in self.pop_inputs.values()])

    
class PopIRMapper:
    def I_to_R(self, I: PopInput) -> PopRegime: pass
    def R_to_I(self, R: PopRegime) -> PopInput: pass
    
class NetIRMapper:
    def __init__(self):
        self.pop_IR_mappers: Dict[str, PopIRMapper] = {}
        
    def set_pop_mapper(self, pop_name: str, mapper: PopIRMapper):
        self.pop_IR_mappers[pop_name] = mapper
        
    def I_to_R(self, I: NetInput) -> NetRegime:
        R = NetRegime()
        for name, I_ in I.pop_inputs.items():
            R.pop_regimes[name] = self.pop_IR_mappers[name].I_to_R(I_)
        return R
    
    def R_to_I(self, R: NetRegime) -> NetInput:
        I = NetInput()
        for name, R_ in R.pop_regimes.items():
            I.pop_inputs[name] = self.pop_IR_mappers[name].R_to_I(R_)
        return I


class NetUCMapper:
    def _Ru_to_Rc(self, Ru: NetRegime) -> NetRegime:
        pass
    
    def _Rc_to_Ru(self, Rc: NetRegime) -> NetRegime:
        pass
    
    def Ru_to_Rc(
            self,
            Ru: Union[NetRegime, list[NetRegime]]
            ) -> Union[NetRegime, list[NetRegime]]:
        """Unconnected -> connected. """
        if isinstance(Ru, NetRegime):
            return self._Ru_to_Rc(Ru)
        else:
            return NetRegimeList([self._Ru_to_Rc(Ru_) for Ru_ in Ru])
        
    def Rc_to_Ru(
            self,
            Rc: Union[NetRegime, list[NetRegime]]
            ) -> Union[NetRegime, list[NetRegime]]:
        """Connected -> unconnected. """
        if isinstance(Rc, NetRegime):
            return self._Rc_to_Ru(Rc)
        else:
            return NetRegimeList([self._Rc_to_Ru(Rc_) for Rc_ in Rc])


@dataclass        
class PopRegimeWC(PopRegime):
    r: float = 0
    
@dataclass
class NetRegimeWC(NetRegime):
    def __init__(self, pop_names: List[str] = None, pop_rates: List[float] = None):
        self.pop_regimes = {}
        if pop_names is not None:
            for pop_name, r in zip(pop_names, pop_rates):
                self.pop_regimes[pop_name] = PopRegimeWC(r=r)
    
    def get_pop_rates_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('r')

@dataclass
class NetRegimeListWC(NetRegimeList):
    def get_pop_rates_mat(self) -> np.ndarray:
        """Returns (pops x regimes) matrix. """
        return self.get_pop_attr_mat('r')


@dataclass
class PopInputWC(PopInput):
    I: float = 0

@dataclass
class NetInputWC(NetInput):
    def get_pop_inputs_vec(self) -> np.ndarray:
        return self.get_pop_attr_vec('I')


@dataclass
class PopParamsWC:
    mult: float = 10
    gain: float = 1
    thresh: float = 1
    
class ModelDescWC(ModelDesc):    
    def __init__(self, num_pops):
        super().__init__()
        self.pops = {}
        for n in range(num_pops):
            self.pops[f'pop{n}'] = PopParamsWC()
        self.conn = 2 * np.random.rand(num_pops, num_pops) - 1
    
    def get_pop_names(self) -> List[str]:
        return list(self.pops.keys())

def wc_gain(x: float, pop: PopParamsWC) -> float:
    return pop.mult / (1 + np.exp(-pop.gain * (x - pop.thresh)))

def wc_gain_inv(r: float, pop: PopParamsWC) -> float:
    return -np.log(pop.mult / r - 1) / pop.gain + pop.thresh

def column(x: np.ndarray) -> np.ndarray:
    x = x.squeeze()
    if x.ndim > 1:
        raise ValueError('Input has more than one non-singleton dimension')
    return x.reshape(-1, 1)

def run_wc_model(
        model: ModelDescWC,
        Iext: NetInputWC,
        niter: int,
        R0: NetRegimeWC = None,
        dr_mult: float = 1
        ) -> Tuple[NetRegimeWC, Dict]:
    npops = len(model.pops)
    # External input
    #ii_ext = Iext.get_pop_inputs_vec().reshape(-1, 1)
    ii_ext = Iext.get_pop_attr_vec('I')
    # Initial rates
    if R0 is None:
        rr = np.zeros(npops)
    else:
        rr = R0.get_pop_rates_vec()
    # Simulation details
    info = {'r_mat': np.zeros((npops, niter)),
            'dr_mat': np.zeros((npops, niter))}
    # Main loop
    rr_new = np.zeros(npops)
    for n in range(niter):
        ii = (model.conn @ column(rr) + ii_ext).flatten()
        for m, pop in enumerate(model.pops.values()):
            rr_new[m] = wc_gain(ii[m], pop)
        drr = rr_new - rr
        rr += dr_mult * drr
        info['r_mat'][:, n], info['dr_mat'][:, n] = rr, drr
    # Result
    R = NetRegimeWC(model.get_pop_names(), rr)
    return R, info


class PopIRMapperWC(PopIRMapper):
    def __init__(self, pop_params: PopParamsWC):
        self.pop_params = pop_params
        
    def I_to_R(self, I: PopInputWC) -> PopRegimeWC:
        r = wc_gain(I.I, self.pop_params)
        return PopRegimeWC(r=r)

    def R_to_I(self, R: PopRegimeWC) -> PopInputWC:
        I = wc_gain_inv(R.r, self.pop_params)
        return PopInputWC(I=I)

class NetIRMapperWC(NetIRMapper):
    def __init__(self, model: ModelDescWC):
        super().__init__()
        for name, pop in model.pops.items():
            self.set_pop_mapper(name, PopIRMapperWC(pop))


def _vec_or_item(x):
    return x if x.size > 0 else x.item()

class NetUCMapperWC(NetUCMapper):
    def __init__(self):
        self.pop_names = []
        self.is_identity = True
        self.fit_par = {'a': [], 'b': [], 'k': []}
    
    #@staticmethod
    def map_func(self, rr_u, a, b, k):
        rr_c = a * np.exp(b * np.array(rr_u)) + k
        rr_c[rr_c < 0] = np.nan
        return _vec_or_item(rr_c)
    
    #@staticmethod
    def map_func_inv(self, rr_c, a, b, k):
        rr_u = np.log((np.array(rr_c) - k) / a) / b
        rr_u[rr_u < 0] = np.nan
        return _vec_or_item(rr_u)
    
    def set_to_identity(self, pop_names: List[str]):
        self.pop_names = pop_names
        self.is_identity = True
    
    def _Ru_to_Rc(self, Ru: NetRegimeWC) -> NetRegimeWC:
        if Ru.get_pop_names() != self.pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        rr_u = Ru.get_pop_rates()
        if self.is_identity:
            rr_c = rr_u
        else:
            rr_c = self.map_func(rr_u, **self.fit_par)
        return NetRegimeWC(self.pop_names, rr_c)
        
    def _Rc_to_Ru(self, Rc: NetRegimeWC) -> NetRegimeWC:
        if Rc.get_pop_names() != self.pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        rr_c = Rc.get_pop_rates_vec()
        if self.is_identity:
            rr_u = rr_c
        else:
            rr_u = self.map_func_inv(rr_c, **self.fit_par)
        return NetRegimeWC(self.pop_names, rr_u)
    
    def fit_from_data(self, Ru: NetRegimeListWC, Rc: NetRegimeListWC):
        if len(Ru) != len(Rc):
             raise ValueError('Ru and Rc should have the same length')
        if Ru.get_pop_names() != Rc.get_pop_names():
            raise ValueError('Ru and Rc entries should have the same pops.')
        self.is_identity = False
        self.pop_names = Ru.get_pop_names()        
        npops = len(self.pop_names)
        for p in ['a', 'b', 'k']:
            self.fit_par[p] = np.zeros(npops)
        def fit_func(rr_u, *args):
            return self.map_func(rr_u, *args)
        rr_u_mat = Ru.get_pop_attr_mat('r')
        rr_c_mat = Rc.get_pop_attr_mat('r')
        for n in range(npops):
            rr_u = rr_u_mat[n, :]
            rr_c = rr_c_mat[n, :]
            b0 = 1.0
            k0 = np.nanmin(rr_c) - 0.1 * np.abs(np.nanmin(rr_c))
            a0 = (np.nanmax(rr_c) - k0) / np.exp(b0 * np.nanmax(rr_u))
            par0 = [a0, b0, k0]
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            try:
                par, _ = curve_fit(fit_func, rr_u, rr_c, p0=par0,
                                   bounds=bounds, nan_policy='omit')
            except Exception as e:
                print(f'Fitting failed ({e})')
                par = [np.nan] * 3
            for m, p in enumerate(['a', 'b', 'k']):
                self.fit_par[p][n] = par[m]
          

# Network of Wilson-Cowan populations
npops = 1
model = ModelDescWC(num_pops=npops)
pop_names = model.get_pop_names()
model.conn[0, 0] = -0.1

# Original target regime (vector of pop. firing rates)
rr_base = np.arange(npops) + 1

# P_FR
pfr_vec = np.linspace(0.1, 1.5, 20)

# Target regimes (base * pfr for each pfr)
R0_lst = NetRegimeListWC(
    [NetRegimeWC(pop_names, rr_base * pfr) for pfr in pfr_vec]
)

# I-R mapper, explicitly uses Wilson-Cowan gain functions of populations
ir_mapper = NetIRMapperWC(model)

# Unconnected-to-connected regime mapper
uc_mapper = NetUCMapperWC()
uc_mapper.set_to_identity(pop_names)

# Params of WC model simulations
sim_par = {'niter': 20, 'dr_mult': 1}

Rc_lst = R0_lst.copy()

for iter_num in range(100):
    print(f'Iter: {iter_num}')
    Rc_lst = R0_lst.copy()
    Rc_prev_lst = Rc_lst.copy()
    Ru_lst = uc_mapper.Rc_to_Ru(Rc_lst)
    for n, Ru in enumerate(Ru_lst):
        Iu = ir_mapper.R_to_I(Ru)
        Rc_new, sim_info = run_wc_model(
            model, Iu, sim_par['niter'], Ru, sim_par['dr_mult']
        )
        Rc_lst[n] = Rc_new
        sim_err = np.abs(sim_info['dr_mat'][:, -1]).max()
        err = np.abs(R0_lst[n].get_pop_rates_vec() - Rc_new.get_pop_rates_vec()).max()
        print(f'Point: {n}, sim_err = {sim_err:.04f}, err = {err:.04f}')
    uc_mapper.fit_from_data(Ru_lst, Rc_lst)
    
    need_plot = 1
    if need_plot:
        plt.figure(112)
        plt.clf()
        ru_mat = Ru_lst.get_pop_attr_mat('r')
        rc_mat = Rc_lst.get_pop_attr_mat('r')
        rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('r')
        iu_mat = np.full_like(ru_mat, np.nan)
        for m in range(ru_mat.shape[1]):
            Ru_ = NetRegimeWC(pop_names, ru_mat[:, m])
            iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_attr_vec('I')
        for n in range(npops):
            rr_u = ru_mat[n, :]
            rr_c = rc_mat[n, :]
            rr_c_prev = rc_prev_mat[n, :]
            ii_u = iu_mat[n, :]
            for x in [rr_c_prev, rr_c, rr_u, ii_u]:
                print(np.round(x, 2))
            plt.subplot(2, npops, n + 1)
            plt.plot(ii_u, rr_u, '.')
            x = np.linspace(np.nanmin(ii_u), np.nanmax(ii_u), 200)
            plt.plot(x, wc_gain(x, model.pops[f'pop{n}']))
            plt.xlabel('Iu')
            plt.ylabel('Ru')
            plt.xlim(-3.5, 0)
            plt.ylim(0, 2)
            plt.title(f'pop = {n}')
            plt.subplot(2, npops, npops + n + 1)
            plt.plot(rr_u, rr_c, '.')
            z = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
            par = {key: val[n] for key, val in uc_mapper.fit_par.items()}
            plt.plot(z, uc_mapper.map_func(z, **par))
            plt.plot(rr_u, rr_c_prev, 'kx')
            plt.xlabel('Ru')
            plt.ylabel('Rc')
            plt.xlim(0, 2)
            plt.ylim(0, 2)
        plt.draw()
        if np.isnan(par['a']):
            break
        if not plt.waitforbuttonpress():
            break
                


# =============================================================================
# n = 0
# R0 = R0_vec[n]
# 
# R_iter = [R0]
# 
# for iter_num in range(10):
#     R = R_iter[-1]
#     I = ir_mapper.R_to_I(R)
#     P = I.apply_to_net_params(P_base)
#     req = SimRequest(cfg, P)
#     req_name = f'req_{n}_{iter_num}'
#     disp.submit_sim_request(req_name, req)
#     sim_res = disp.get_sim_result(req_name)
#     R_next = NetRegime.calc_from_sim_result(sim_res)
#     R_iter.append(R_next)
# =============================================================================




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


# =============================================================================
# class NetParams:
#     def __init__(self): pass
# 
# class SimConfig:
#     def __init__(self): pass
# 
# class ModelDescNP(ModelDesc):
#     def __init__(self):
#         self.net_params = NetParams()
#         self.sim_cfg = SimConfig()
# 
# 
# R_base = NetRegime('R_base.json')
# 
# pfr_vec = np.linspace(0, 1, 10)
# R0_vec = [net_regime_from_pfr(R_base, pfr) for pfr in pfr_vec]
# 
# ir_mapper = IRMapper()
# 
# P_base = NetParams()
# cfg = SimConfig()
# 
# disp = SimRequestDispatcher()
# 
# n = 0
# R0 = R0_vec[n]
# 
# R_iter = [R0]
# 
# for iter_num in range(10):
#     R = R_iter[-1]
#     I = ir_mapper.R_to_I(R)
#     P = I.apply_to_net_params(P_base)
#     req = SimRequest(cfg, P)
#     req_name = f'req_{n}_{iter_num}'
#     disp.submit_sim_request(req_name, req)
#     sim_res = disp.get_sim_result(req_name)
#     R_next = NetRegime.calc_from_sim_result(sim_res)
#     R_iter.append(R_next)
# 
# 
# class SimRequest:
#     def __init__(self, cfg: SimConfig, net_params: NetParams):
#         self.cfg = cfg
#         self.net_params = net_params
# 
# class SimResult:
#     def __init__(self): pass
#         
# class SimRequestDispatcher:
#     def submit_sim_request(self, name, req: SimRequest): pass
#     def get_sim_result(self, name) -> SimResult: pass
# =============================================================================
