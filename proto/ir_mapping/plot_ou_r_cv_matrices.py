from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


# Contour function: y = k * x
def contour_func_const_ratio(x, y, k):
    return y - k * x

# Find contours: func(x, y) = level
def find_contours(func, x, y, level=0, **kwargs):
    xx, yy = np.meshgrid(x, y)
    z = func(xx, yy, **kwargs)
    C_idx = measure.find_contours(z, level)
    C = []
    for n, c_idx in enumerate(C_idx):
        c_idx = c_idx.round().astype(int)
        c = np.full_like(c_idx, np.nan, dtype=float)
        c[:, 0] = y[c_idx[:, 0]]
        c[:, 1] = x[c_idx[:, 1]]
        C.append(c)
        C_idx[n] = c_idx
    return C, C_idx


dirpath_base = Path('D:\\WORK\\Salvador\\repo\\model_tuner\\test_data\\a1_ou_unconn\\scott_2025_02_28')
fpath_in = dirpath_base / 'OUmapping_0228.pkl'

dirpath_out = dirpath_base / 'plots'
dirpath_out.mkdir(exist_ok=True)

with open(fpath_in, 'rb') as file:
    data = pickle.load(file)

pop_names = list(data['rate'].keys())

#pop_names = ['IT3']

for m, pop_name in enumerate(pop_names):

    print(f'{m} {pop_name}...')

    R = data['rate'][pop_name]
    CV = data['isicv'][pop_name]

    Xvis = {'Firing rate': R, 'CV': CV}

    # Coordinates
    ouamp_vec = R.columns.values * 100
    oustd_vec = R.index.values * 100

    # Find contour: oustd / ouamp = const
    k = 0.4
    C, C_idx = find_contours(contour_func_const_ratio, ouamp_vec, oustd_vec, k=k)
    C, C_idx = C[0], C_idx[0]

    plt.figure(111, figsize=(10, 8))
    plt.clf()
    plt.ion()

    for n, (xname, X) in enumerate(Xvis.items()):

        plt.subplot(2, 2, 2 * n + 1)
        par = {}
        if xname == 'CV':
            par |= {'vmin': 0, 'vmax': 2}
        ext = (ouamp_vec[0], ouamp_vec[-1], oustd_vec[0], oustd_vec[-1])
        plt.imshow(X, aspect='auto', cmap='viridis', origin='lower', extent=ext, **par)
        plt.colorbar()
        for k in range(0, len(oustd_vec), 4):
            plt.plot((ouamp_vec.min(), ouamp_vec.max()), (oustd_vec[k], oustd_vec[k]), '--')
        #plt.plot(ouamp_vec, 0.4 * ouamp_vec, 'k--', label='std = 0.4 * amp')
        plt.plot(C[:, 1], C[:, 0], 'k--', label=f'std = {k} * amp')
        #plt.legend()
        if n == 1:  plt.xlabel('ouamp * 100')
        plt.ylabel('oustd * 100')
        plt.title(f'{pop_name}: {xname}')
        plt.xlim(ouamp_vec[0], ouamp_vec[-1])
        plt.ylim(oustd_vec[0], oustd_vec[-1])

        plt.subplot(2, 2, 2 * n + 2)
        for k in range(0, len(oustd_vec), 4):
            plt.plot(ouamp_vec, X.iloc[k, :], '--') #, label=f"std={oustd}")
        ouamp_idx = C_idx[:, 1]
        oustd_idx = C_idx[:, 0]
        xx = [X.iloc[oustd_idx[n], ouamp_idx[n]]
              for n in range(len(ouamp_idx))]
        plt.plot(ouamp_vec[ouamp_idx], xx, 'k', lw=2)
        if n == 1: plt.xlabel('ouamp * 100')
        plt.ylabel(xname)
        plt.title(f'{pop_name}: {xname}')
        if n % 2 == 1:  plt.ylim(0, 2)

    plt.draw()
    plt.show()

    fpath_out = dirpath_out / f'{m}_{pop_name}.png'
    plt.savefig(fpath_out, dpi=300)

    #break

#input('Press any key...')