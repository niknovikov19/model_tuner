from dataclasses import dataclass

from model_tuner.opt_base import PopInput


@dataclass
class PopInput1D(PopInput):
    value: float

@dataclass
class PopInputRate(PopInput1D):
    def __init__(self, rate: float):
        super().__init__(value=rate)
        
    @property
    def rate(self) -> float:
        return self.value

    @rate.setter
    def rate(self, val: float):
        self.value = val


R = PopInputRate(rate=10)
print(R)


# =============================================================================
# @dataclass
# class PopInputRate(PopInput1D):
#     @property
#     def rate(self) -> float:
#         return self.value
# 
#     @rate.setter
#     def rate(self, val: float):
#         self.value = val
# 
# @dataclass
# class PopRegimeRate(PopRegime1D):
#     @property
#     def rate(self) -> float:
#         return self.value
# 
#     @rate.setter
#     def rate(self, val: float):
#         self.value = val
# =============================================================================
