"""
This program is written to plot regs.
"""
# imports python standard libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np

font = {'family':'sans Serif', 'weight':'normal', 'size':16} # 'DejaVu Sans'
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'najmeh.kaffashzadeh@gmail.com'


if __name__ == '__main__':
    vals1 = [[-0.39, 0.63, -0.08, 0.08], [-0.11, 0.71, 0.02, 0.04], [0.15, 0.81, 0.14, 0.05]]
    yerr1 = [[0.01, 0.02, 0.01, 0.01], [0.02, 0.02, 0.1, 0.1], [0.02,0.02,0.01,0.01]]
    vals2 = [[-0.34,0.23,0.08,0.01],[0.25,0.21,-0.09,-0.23],[-0.16,0.09,0.05,0.3]]
    yerr2 = [[0.02,0.02,0.02,0.02],[0.02,0.02,0.02,0.02],[0.02,0.02,0.02,0.02]]
    fig, ax=plt.subplots(nrows=1,ncols=2, figsize=(13,5))
    x=[1,2,3,4]
    ax[0].errorbar(x, [i for i in vals1[0]], yerr=yerr1[0], fmt='o', color='k')
    ax[0].errorbar(x, [i for i in vals1[1]], yerr=yerr1[1], fmt='o', color='r')
    ax[0].errorbar(x,[i for i in vals1[2]], yerr=yerr1[2], fmt='o', color='dodgerblue')
    ax[0].set_ylim(-1,1)
    ax[0].axhline(y=0, color='k', linestyle=':')
    ax[0].set_xticks(np.arange(1, 5, 1))
    ax[0].set_xticklabels(['a$_{1}$', 'a$_{2}$', 'a$_{3}$', 'a$_{4}$'], color='k')
    ax[0].text(0.79, 0.95, 'R$^{2}$ = 0.67',
               horizontalalignment='left', transform=ax[0].transAxes, color='k')
    ax[0].text(0.79, 0.9, 'R$^{2}$ = 0.68',
               horizontalalignment='left', transform=ax[0].transAxes, color='r')
    ax[0].text(0.79, 0.85, 'R$^{2}$ = 0.56',
               horizontalalignment='left', transform=ax[0].transAxes, color='dodgerblue')
    ax[1].errorbar(x, [i for i in vals2[0]], yerr=yerr2[0], fmt='o', color='k')
    ax[1].errorbar(x, [i for i in vals2[1]], yerr=yerr2[1], fmt='o', color='r')
    ax[1].errorbar(x, [i for i in vals2[2]], yerr=yerr2[2], fmt='o', color='dodgerblue')
    ax[1].set_ylim(-1, 1)
    ax[1].axhline(y=0, color='k', linestyle=':')
    ax[1].text(0.79, 0.95, 'R$^{2}$ = 0.20',
                horizontalalignment='left', transform=ax[1].transAxes, color='k')
    ax[1].text(0.79, 0.9, 'R$^{2}$ = 0.23' ,
               horizontalalignment='left', transform=ax[1].transAxes, color='r')
    ax[1].text(0.79, 0.85, 'R$^{2}$ = 0.09' ,
               horizontalalignment='left', transform=ax[1].transAxes, color='dodgerblue')
    # ax[1].text(0.7, 0.8, 'R $ˆ{2}$ = 0.09' + str('%.2f' % 0.5),
    #            horizontalalignment='left', transform=ax[1].transAxes, color='dodgerblue')
    ax[1].set_xticks(np.arange(1, 5, 1))
    ax[1].set_xticklabels(['a$_{1}$', 'a$_{2}$', 'a$_{3}$', 'a$_{4}$'], color='k')
    ax[0].set_title('(a)')
    ax[1].set_title('(b)')
    fig.tight_layout()
    plt.savefig('coef.jpg', bbox_inches='tight')
    plt.close()
    exit()















    fig = plt.figure(figsize=(13, 8))# constrained_layout=True)
    gs0 = gridspec.GridSpec(6,4)#wspace=0.05, hspace=0.05)
    #gs0 = gridspec.GridSpecFromSubplotSpec(6,4, subplot_spec=gs[0],wspace=0.2, hspace=0.2)
    ax01 = fig.add_subplot(gs0[0:2, 0:2])
    ax02 = fig.add_subplot(gs0[2:5, 0:2])
    ax02.set_xlabel('a')
    ax03 = fig.add_subplot(gs0[5, :])
    ax03.set_title('s')
    ax04 = fig.add_subplot(gs0[0:2, 2:3])
    ax04.set_xlabel('a')
    ax05 = fig.add_subplot(gs0[2:5, 2:3])
    ax05.set_xlabel('a')
    ax06 = fig.add_subplot(gs0[0:5, 3])
    ax06.set_xlabel('a')
    gs0.tight_layout(fig)#, rect=[None, None, 0.5, None])
    plt.savefig('tet.png', bbox_inches='tight')
    plt.close()
    exit()
    plt.show()
    exit()