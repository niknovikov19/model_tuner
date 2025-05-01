import matplotlib
import matplotlib.pyplot as plt

from netpyne import sim
import pandas as pd


fpath_sim = (
    r'D:\WORK\Salvador\repo\model_tuner\models\A1_OUinp\A1_OUinp\simulations'
    r'\test_opt_A1_batch_qsub\test_2_pfr=(0.4_1.0_4)_wmult=0.005_alpha=0.2\req_19_3_data.pkl'
)

sim.initialize()
sim.loadAll(fpath_sim, instantiate=False)

matplotlib.use('Qt5Agg', force=True)

#layer = '3'
#pops_vis = [pop + layer for pop in ['IT', 'PV', 'SOM', 'VIP', 'NGF']]
pops_vis = ['TC', 'TCM', 'HTC', 'TI' ,'TIM', 'IRE', 'IREM']

plt.ion()
sim.analysis.plotRaster(orderInverse=True, include=pops_vis, showFig=True)
plt.show()

input("Press any key to continue...")
