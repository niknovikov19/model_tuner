import matplotlib
import matplotlib.pyplot as plt

from netpyne import sim
import pandas as pd


fpath_sim = (
    r'D:\WORK\Salvador\repo\model_tuner\models\A1_OUinp\A1_OUinp\simulations'
    r'\test_opt_A1_batch_qsub\test_3_pfr=(0.4_1.0_4)_wmult=0.02_alpha=1_autosz\req_11_3_data.pkl'
)

fpath_targets = (
    r'D:\WORK\Salvador\repo\model_tuner\test_data\main\test_opt_A1_hpc_batch_qsub'
    r'\experiments\test_3_pfr=(0.4_1.0_4)_wmult=0.02_alpha=1_autosz\target_rates.csv'
)

df = pd.read_csv(fpath_targets)
target_rates = dict(zip(df['pop_name'], df['target_rate']))
target_rates

sim.initialize()
sim.loadAll(fpath_sim, instantiate=False)

matplotlib.use('Qt5Agg', force=True)

#pops_vis = ['IT2']
#pops_vis = ['IT2', 'IT3', 'PV4']

pops_vis = list(sim.net.allPops.keys())

plt.ion()

#sim.analysis.plotRaster(orderInverse=True, include=include, showFig=True)
sim.analysis.plotSpikeStats(include=pops_vis, timeRange=(500, 3000), stats=['rate'])

npops = len(pops_vis)
for n, pop in enumerate(pops_vis):
    r = target_rates[pop]
    y = npops - n
    d = 0.4
    #plt.plot([0, 10], [y, y], '--')
    plt.plot([r, r], [y - d, y + d], 'r', linewidth=5)

plt.show()

input("Press any key to continue...")
