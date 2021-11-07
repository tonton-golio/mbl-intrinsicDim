import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_eigencomponentDominance(below_lims, maxs, ws,
                 lims = np.logspace((1-8),0,8),
                 num_seeds =10, L=8,
                 colors = 'orange, lightblue, salmon, yellowgreen, grey, purple'.split(', ')
                ):
    # Plot 1: proportion below threshhold
    mean_below = np.mean(below_lims,axis=2).T[::-1]
    fig, ax  = plt.subplots(2,1, sharex=True, 
                           gridspec_kw={'height_ratios':[2,1]},
                           figsize=(6,4))
    #print(len(mean_below))
    for i, color in zip(range(len(mean_below)), colors):
        #print(i)
        try:
            ax[0].fill_between(ws, mean_below[i], mean_below[i+1],
                         label="{:.0e}".format(lims[::-1][i]),
                         color=color, alpha=.3)
        except IndexError:
            ax[0].fill_between(ws, 0, mean_below[i],
                         label="{:.0e}".format(lims[::-1][i]),
                         color=color, alpha=.3)
		
    
    # Plot 2: Maxs
    for index, i in enumerate(maxs):
        ax[1].scatter([ws[index]]*num_seeds, 1-i, c='b', alpha=2/num_seeds)
        ax[1].scatter([ws[index]], 1-np.mean(i), c='r', alpha=0.9)

    ax[1].grid()
    # Labels and such
    ax[0].legend(bbox_to_anchor=(.23, .75), fontsize=11)
    ax[0].set_ylabel('Proportion of $|\kappa|<\zeta$ ', fontsize=14)
    ax[1].legend(["point", "mean"],#bbox_to_anchor=(0.2, .25),
    	facecolor='white', framealpha=1,
        fontsize=12)
    plt.xlabel('Disorder strength, $W$', fontsize=14)
    ax[1].set_ylabel('$1-max(|\kappa|)$', fontsize=14)    
    #plt.suptitle('Eigencomponent, $\kappa$, dominance', fontsize=17)

def plot_2nn(x,y,d):
    plt.scatter(x,y, c='purple', alpha=0.5)
    plt.plot(x,x*d, c='orange', ls='-')
    plt.title('2NN output for single realization', fontsize=16)
    plt.xlabel('$\ln(\mu)$', fontsize=14)
    plt.ylabel('$-\ln(1 - F(\mu))$', fontsize=14)
    plt.text(np.mean(x)*.75, np.mean(y)*1.5, s='$D_{int}$'+'={}'.format(round(d,1)), fontsize=14)
    plt.grid()



def plot_2nn_imshow(data, ws):
    imshow = plt.imshow(data,
            aspect=.5*data.shape[1]/data.shape[0],cmap = 'inferno', norm=LogNorm())


    plt.colorbar(aspect=2.5, )
    plt.title('$\mathcal{D}_{int}$ for many realizations', fontsize=14)
    plt.ylabel('Seed', fontsize=12)
    plt.xlabel('Disorder strength, $W$', fontsize=12)
    plt.xticks(np.linspace(0,len(ws)-1,5), np.linspace(min(ws), max(ws),5))
    ax = imshow.axes
    ax.invert_yaxis()

    

def plot_2nn_avgs(ID_avgs, Ls, ws, num_seeds):
    for index, L in enumerate(Ls):
        plt.plot(ws,ID_avgs[index], label='L={}'.format(L))
        
    plt.legend()
    plt.xlabel('Disorder Strength', fontsize=13)
    plt.ylabel('weighted average ID', fontsize=13)
    plt.title('ID from 2nn weigted average, seeds={}, ws={}'.format(L,num_seeds,len(ws)), fontsize=16)
    plt.show()


def plot_plateau(plateau_dict):
    plt.figure(figsize=(6,4),tight_layout=True)

    for L in plateau_dict:
        if L != 'params':
            x = sorted(plateau_dict[L].keys())
            y = [plateau_dict[L][i]['id'] for i in x]
            yerr = [plateau_dict[L][i]['std'] for i in x]
            plt.errorbar(x,y,yerr=yerr,
                alpha = 0.6,
                label=f"{L}; {plateau_dict['params'][L]}")

    plt.xlabel('Proportion of eigenstates considered', fontsize=13)
    plt.ylabel('Intrinsic dimension', fontsize=13)

    #plt.title('Plateauing_new, W={}'.format(plateau_dict['params']['W']), fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=4, fancybox=True, shadow=True)
    plt.grid()
    return plateau_dict['params']