from typing import Dict, Any

#
# Statistic distributions
#
# from random import Random
# 
# class Distrib:
#     @property
#     def value(self): pass
# 
# 
# class ConstantDistrib(Distrib):
#     def __init__(self, v):
#         self._value = v
#     @property
#     def value(self): return self._value
# 
# 
# class UniformDistrib(Distrib):
#     def __init__(self, a, b):
#         self._value = 1/(b-a)
#     @property
#     def value(self): return self._value
# 
# 
# class PoissonDistrib(Distrib):
#     def __init__(self, lmbda, seed=None):
#         self._lmbda = lmbda
#         self._rnd = Random(seed)
# 
#     @property
#     def value(self):
#         x = self._rnd.random()
#         return self._rnd.

class Distrib:

    @property
    def value(self) -> float: pass

    @property
    def mean(self) -> float: pass

    @property
    def asdict(self) -> Dict[str, Any]: pass

    @staticmethod
    def fromdict(d: dict):
        distrib = None
        type = d["type"]
        if type == "NormalDistrib":
            distrib = NormalDistrib()
        elif type == "ConstantDistrib":
            distrib = ConstantDistrib()
        elif type == "PoissonDistrib":
            distrib = PoissonDistrib()
        else:
            raise Exception("Undefined distribution " + distrib)

        return distrib.fromdict(d)
# end


class NormalDistrib(Distrib):
    def __init__(self, mean: float = 0, sdev: float = 1):
        self.mean: float = mean
        self.sdev: float = sdev

    @property
    def value(self) -> float:
        return norm.rvs(self.mean, self.sdev)

    @property
    def asdict(self):
        return {
            "type": "NormalDistrib",
            "mean": self.mean,
            "sdev": self.sdev
        }

    def fromdict(self, d):
        self.mean = d["mean"]
        self.sdev = d["sdev"]
        return self
# end


class ConstantDistrib(Distrib):
    def __init__(self, v: float = 0):
        self._value: float = v

    @property
    def value(self) -> float:
        return self._value

    @property
    def mean(self) -> float:
        return self._value

    @property
    def asdict(self):
        return {
            "type": "ConstantDistrib",
            "value": self._value
        }

    def fromdict(self, d: dict):
        self._value = d["value"]
        return self
# end


class PoissonDistrib(Distrib):

    def __init__(self, mean: float = 0, min: float = 0):
        self.mean: float = mean
        self.min: float = min
        self.mu: float = mean - min

    @property
    def value(self) -> float:
        return poisson.rvs(self.mu, loc=self.min)

    @property
    def mean(self) -> float:
        return self.mean

    @property
    def asdict(self) -> Dict[str,Any]:
        return {
            "type": "PoissonDistrib",
            "mean": self.mean,
            "min": self.min,
            "mu": self.mu
        }

    def fromdict(self, d: dict):
        self.mean = d["mean"]
        self.min = d["min"]
        self.mu = d["mu"]
        return self
# end
