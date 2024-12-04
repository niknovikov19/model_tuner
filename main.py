from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _vec_or_item(x):
    return x if x.size > 0 else x.item()

def column(x: np.ndarray) -> np.ndarray:
    x = x.squeeze()
    if x.ndim > 1:
        raise ValueError('Input has more than one non-singleton dimension')
    return x.reshape(-1, 1)


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
            Ru: Union[NetRegime, NetRegimeList]
            ) -> Union[NetRegime, NetRegimeList]:
        """Unconnected -> connected. """
        if isinstance(Ru, NetRegime):
            return self._Ru_to_Rc(Ru)
        else:
            return NetRegimeList([self._Ru_to_Rc(Ru_) for Ru_ in Ru])
        
    def Rc_to_Ru(
            self,
            Rc: Union[NetRegime, NetRegimeList]
            ) -> Union[NetRegime, NetRegimeList]:
        """Connected -> unconnected. """
        if isinstance(Rc, NetRegime):
            return self._Rc_to_Ru(Rc)
        else:
            return NetRegimeList([self._Rc_to_Ru(Rc_) for Rc_ in Rc])
        
    def fit_from_data(self, Ru: NetRegimeList, Rc: NetRegimeList): pass


@dataclass        
class PopRegimeWC(PopRegime):
    r: float = 0
    
@dataclass
class NetRegimeWC(NetRegime):
    def __init__(self, pop_names: List[str] = None,
                 pop_rates: List[float] = None):
        self.pop_regimes = {}
        if pop_names is not None:
            for pop_name, r in zip(pop_names, pop_rates):
                self.pop_regimes[pop_name] = PopRegimeWC(r=r)
    
    def get_pop_rate(self, pop_name: str) -> float:
        return self.pop_regimes[pop_name].r
    
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


class MapFunc1D:
    def __init__(self):
        self.par = {name: np.nan for name in self.get_par_names()}
    
    @staticmethod
    def get_par_names() -> List[str]: pass

    def get_par_vals(self) -> List:
        return [self.par[name] for name in self.get_par_names()]
        
    @staticmethod
    def f(x: Union[float, np.ndarray],
          *args, **kwargs
          ) -> Union[float, np.ndarray]:
        pass
    
    @staticmethod
    def f_inv(x: Union[float, np.ndarray],
              *args, **kwargs
              ) -> Union[float, np.ndarray]:
        pass
    
    def apply(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.f(x, **self.par)
    
    def apply_inv(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.f_inv(y, **self.par)
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]: pass
        
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple: pass
    
    def fit(self, xx: np.ndarray, yy: np.ndarray, from_prev=False):
        if from_prev:
            par0 = self.get_par_vals()
        else:
            par0 = self._get_first_fit_guess(xx, yy)
        bounds = self._get_fit_bounds()
        try:
            par, _ = curve_fit(self.f, xx, yy, p0=par0,
                               bounds=bounds,
                               nan_policy='omit')
            self.par = {name: par[n]
                        for n, name in enumerate(self.get_par_names())}
        except Exception as e:
            print(f'Fitting failed ({e})')
            self.par = {name: np.nan for name in self.get_par_names()}

class MapFunc1DExp(MapFunc1D):    
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'k']

    @staticmethod
    def f(x: Union[float, np.ndarray],
          a, b, k
          ) -> Union[float, np.ndarray]:
        y = a * np.exp(b * x) + k
        #y[y < 0] = np.nan
        return y
    
    @staticmethod
    def f_inv(y: Union[float, np.ndarray],
              a, b, k
              ) -> Union[float, np.ndarray]:
        x = np.log((y - k) / a) / b
        #x[x < 0] = np.nan
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        return [0, 0, -100], [np.inf, np.inf, np.inf]
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        b0 = 1.0
        k0 = np.nanmin(yy) - 0.1 * np.abs(np.nanmin(yy))
        a0 = (np.nanmax(yy) - k0) / np.exp(b0 * np.nanmax(xx))
        return a0, b0, k0
    
class MapFunc1DSigmoid(MapFunc1D):
    @staticmethod
    def get_par_names() -> List[str]:
        return ['a', 'b', 'c', 'k']
    
    @staticmethod
    def f(x: Union[float, np.ndarray],
          a, b, c, k
          ) -> Union[float, np.ndarray]:
        y = c + a / (1 + np.exp(-k * (x - b)))
        return y
    
    @staticmethod
    def f_inv(y: Union[float, np.ndarray],
              a, b, c, k
              ) -> Union[float, np.ndarray]:
        x = b - np.log(a / (y - c) - 1) / k
        return x
    
    @staticmethod
    def _get_fit_bounds() -> Tuple[List[float], List[float]]:
        bounds = {
            'a': (0.1, 10),
            'b': (-10, 20),
            'c': (-10, 20),
            'k': (0.1, 10)
        }
        low = [bounds[p][0] for p in ['a', 'b', 'c', 'k']]
        high = [bounds[p][1] for p in ['a', 'b', 'c', 'k']]
        return low, high
    
    @staticmethod
    def _get_first_fit_guess(xx: np.ndarray, yy: np.ndarray) -> Tuple[float]:
        k0 = 1
        c0 = np.nanmin(yy)
        a0 = np.nanmax(yy) - c0
        th = c0 + a0 / 2
        n1 = np.argmax(yy[yy < th])
        n2 = np.argmin(yy[yy > th])
        x1, y1 = xx[n1], yy[n1]
        x2, y2 = xx[n2], yy[n2]
        b0 = (x1 + x2) / 2
        return a0, b0, c0, k0


class NetUCMapperWC(NetUCMapper):
    def __init__(
            self,
            pop_names: List[str],
            map_type: Literal['exp', 'sigmoid'] = 'exp'):
        self._pop_names = pop_names
        self._is_identity = True
        self._map_funcs = {}
        for pop in self._pop_names:
            if map_type == 'exp':
                self._map_funcs[pop] = MapFunc1DExp()
            elif map_type == 'sigmoid':
                self._map_funcs[pop] = MapFunc1DSigmoid()
            else:
                raise ValueError(f'Unknown map type: {map_type}')
    
    def set_to_identity(self):
        self._is_identity = True
    
    def _Ru_to_Rc(self, Ru: NetRegimeWC) -> NetRegimeWC:
        if Ru.get_pop_names() != self._pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        if self._is_identity:
            Rc = {pop: Ru.get_pop_rate(pop) for pop in self._pop_names}
        else:
            Rc = {pop: self._map_funcs[pop].apply(Ru.get_pop_rate(pop))
                  for pop in self._pop_names}
        return NetRegimeWC(pop_regimes=Rc)
        
    def _Rc_to_Ru(self, Rc: NetRegimeWC) -> NetRegimeWC:
        if Rc.get_pop_names() != self._pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        if self._is_identity:
            Ru = {pop: Rc.get_pop_rate(pop) for pop in self._pop_names}
        else:
            Ru = {pop: self._map_funcs[pop].apply_inv(Rc.get_pop_rate(pop))
                  for pop in self._pop_names}
        return NetRegimeWC(self._pop_names, Ru.values())
    
    def fit_from_data(self, Ru: NetRegimeListWC, Rc: NetRegimeListWC):
        if len(Ru) != len(Rc):
             raise ValueError('Ru and Rc should have the same length')
        if Ru.get_pop_names() != self._pop_names:
            raise ValueError('Ru should have the same pops. as the mapper')
        if Rc.get_pop_names() != self._pop_names:
            raise ValueError('Rc should have the same pops. as the mapper')
        rr_u_mat = Ru.get_pop_attr_mat('r')
        rr_c_mat = Rc.get_pop_attr_mat('r')
        from_prev = not self._is_identity
        for n, pop in enumerate(self._pop_names):
            self._map_funcs[pop].fit(rr_u_mat[n, :], rr_c_mat[n, :], from_prev)
        self._is_identity = False
          

def create_test_model_1pop():
    model = ModelDescWC(num_pops=1)
    model.conn[0, 0] = -0.1
    return model

def create_test_model_2pop():
    model = ModelDescWC(num_pops=2)
    model.conn = np.array([[0.1, -0.2], [0.2, -0.1]])
    return model

# Network of Wilson-Cowan populations
#model = create_test_model_1pop()
model = create_test_model_2pop()

pop_names = model.get_pop_names()
npops = len(pop_names)

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
#uc_mapper = NetUCMapperWC(pop_names, 'exp')
uc_mapper = NetUCMapperWC(pop_names, 'sigmoid')
uc_mapper.set_to_identity()

# Params of WC model simulations
sim_par = {'niter': 20, 'dr_mult': 1}

need_plot_iter = 0
need_plot_res = 1

n_iter = 20

for iter_num in range(n_iter):
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
        n_nan = np.sum(np.isnan(Rc_new.get_pop_rates_vec()))
        print(f'Point: {n}, sim_err = {sim_err:.02f}, nan = {n_nan}, '
              f'err = {err:.04f}')
    uc_mapper.fit_from_data(Ru_lst, Rc_lst)
    
    if need_plot_iter or (need_plot_res and (iter_num == (n_iter - 1))):
        plt.figure(112)
        plt.clf()
        ru_mat = Ru_lst.get_pop_attr_mat('r')
        rc_mat = Rc_lst.get_pop_attr_mat('r')
        rc_prev_mat = Rc_prev_lst.get_pop_attr_mat('r')
        iu_mat = np.full_like(ru_mat, np.nan)
        for m in range(ru_mat.shape[1]):
            Ru_ = NetRegimeWC(pop_names, ru_mat[:, m])
            iu_mat[:, m] = ir_mapper.R_to_I(Ru_).get_pop_attr_vec('I')
        for n, pop in enumerate(pop_names):
            rr_u = ru_mat[n, :]
            rr_c = rc_mat[n, :]
            rr_c_prev = rc_prev_mat[n, :]
            ii_u = iu_mat[n, :]
            #for x in [rr_c_prev, rr_c, rr_u, ii_u]:
            #    print(np.round(x, 2))
            plt.subplot(2, npops, n + 1)
            plt.plot(ii_u, rr_u, '.')
            ii_u_ = np.linspace(np.nanmin(ii_u), np.nanmax(ii_u), 200)
            plt.plot(ii_u_, wc_gain(ii_u_, model.pops[pop]))
            plt.xlabel('Iu')
            plt.ylabel('Ru')
            rvis_max = rr_base[n] * pfr_vec.max() * 1.2
            plt.xlim(-3.5, 0)
            plt.ylim(0, rvis_max)
            plt.title(f'pop = {pop}')
            plt.subplot(2, npops, npops + n + 1)
            plt.plot(rr_u, rr_c, '.')
            rr_u_ = np.linspace(np.nanmin(rr_u), np.nanmax(rr_u), 200)
            plt.plot(rr_u_, uc_mapper._map_funcs[pop].apply(rr_u_))
            plt.plot(rr_u, rr_c_prev, 'kx')
            plt.xlabel('Ru')
            plt.ylabel('Rc')
            plt.xlim(0, rvis_max)
            plt.ylim(0, rvis_max)
        plt.draw()
        if need_plot_iter:
            if not plt.waitforbuttonpress():
                break
