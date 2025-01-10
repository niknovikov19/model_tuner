from ..inputs import PopInputWC
from ..regimes import PopRegimeWC

from ..wc import PopParamsWC, ModelDescWC, wc_gain, wc_gain_inv

from ..ir_mappers import PopIRMapper, NetIRMapper


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