from defs_base import NetInput, NetRegime, NetRegimeList
from mappers_base import NetIRMapper, NetUCMapper


# Original baseline regime
R0 = NetRegime()

# Target regime list (for the connected network): scaled versions of R0
Rc_lst_target = NetRegimeList()
# Rc_lst_target = R0 * [0.1, 0.2, ..., 1.5]

# Input-to-output mapper (input -> regime)
ir_mapper = NetIRMapper()

# Unconnected-to-connected regime mapper
uc_mapper = NetUCMapper()
# uc_mapper.set_to_identity()

# Run simulation for the connected network with the external input I,
# and get the resulting regime
def run_model(I: NetInput) -> NetRegime:
    pass

# Allocate temporary lists for unconnected and conneted network regimes
Ru_lst = NetRegimeList([])
Rc_lst = NetRegimeList([])

# Main cycle
for iter_num in range(10):
    # For each target regime in the list
    for n, Rc in enumerate(Rc_lst_target):
        # Predict the unconnected network regime (Ru) that corresponds to
        # the given connected network regime (Rc) under the same external input,
        # using the current estimation of Ru-Rc mapping
        Ru = uc_mapper.Rc_to_Ru(Rc)
        
        # Map the regime Ru to the input Iu that will provide this regime
        # in the unconnected network (Ru-Iu mapping is known in advance)
        Iu = ir_mapper.R_to_I(Ru)
        
        # Simulate the connected network with the external input Iu
        Rc_sim = run_model(Iu)
        
        # Store Ru and Rc
        Ru_lst[n], Rc_lst[n] = Ru, Rc_sim
    
    # Update the estimate of the Ru-Rc mapping
    uc_mapper.fit_from_data(Ru_lst, Rc_lst)
