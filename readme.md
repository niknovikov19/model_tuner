**PopRegime** class is an abstraction for a group of population activity properties. Actual properties are to be specified in child classes. In the simplest case, population activity is characterized by a single property: firing rate.

**NetRegime** is a description of network activity. It contains several **PopRegime** objects, one for each population.

**NetRegimeList** is a list of **NetRegime** objects. For convenience, it is defined as a separatee class.

**PopInput** is an abstraction for a group of population input properties. Actual properties are to be specified in child classes. For example, an OU input is characterized by two properties: mean and std.

**NetInput** is a description of network input. It contains several **PopInput** objects, one for each population.

Example:
```
@dataclass
class PopRegimeRate(PopRegime):
	r: float

@dataclass
class PopInputOU(PopInput):
	mean: float
	std: float
```

**PopIRMapper** class handles a mapping between population input (**PopInput**) and its regime (**PopRegime**). This class only provides an interface; the actual mapping code should be implemented in a child class. E.g., it could be a mapping between OU input properties (mean and std) and firing rate.
The interface consists of two mapping functions - forward and inverse (should be implemented in a child class):
```
class PopIRMapper:
    def I_to_R(self, I: PopInput) -> PopRegime
    def R_to_I(self, R: PopRegime) -> PopInput
```

**NetIRMapper** class handles a mapping between network input (**NetInput**) and its regime (**NetRegime**). It contains an instance of **PopIRMapper** class for each network population. The main functionality is already implemented in this class, so there is no need to re-implement it in children.
The interface is:
```
class NetIRMapper:
	def I_to_R(self, I: NetInput) -> NetRegime
	def R_to_I(self, R: NetRegime) -> NetInput
```

