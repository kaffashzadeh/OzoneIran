"""
This program is written to analysis and plot the ozone data time series.
"""
# imports python standard libraries
import os
import sys
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

# import local libraries
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir + '/../../../../TimeSeriesSpectralAnalysis/src/')
os.sys.path.insert(0, parentdir)

font = {'family':'sans Serif', 'weight':'normal', 'size':16} # 'DejaVu Sans'
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'n.kaffashzadeh@ut.ac.ir'


class Assess:

    name = 'Assess the time series'

    def __init__(self):
        """
        This initializes the variables.
        """
        self.df_obs = None

    def read_obs_data(self):
        """
        It reads the observed data form the given file.
        """
        return pd.read_csv(sys.path[0] + '/../data/df_o3_ir_spec.csv',
                            parse_dates=True, index_col=0, header=[0,1,2])

    def calc_var(self, df=None):
        """
        It calculates the variance.

        Args:
            df(pd.DataFrame): data series
        """
        return df.var(axis=0)

    def calc_corr(self, df1=None, df2=None):
        """
        It calculates the correlation between two series.

        Args:
            df1(pd.DataFrame): the first data series
            df2(pd.DataFrame): the second data series
        """
        return df1.corr(df2)

    def calc_cov(self, df1=None, df2=None):
        """
        It calculates the covariance between two series.

        Args:
           df1(pd.DataFrame): the first data series
           df2(pd.DataFrame): the second data series
        """
        return self.calc_corr(df1=df1, df2=df2) * \
               np.sqrt(self.calc_var(df=df1) * self.calc_var(df=df2))

    def plot_time_series(self, df=None, ax=None):
        """"
        It plots the time series of the dataframe.
        """
        df.plot(ax=ax)
        plt.xlabel('date-time')
        plt.ylabel('value')

    def save_fig(self, fn=None):
        """
        It save figure.

        Args:
            fn(str): file name
        """
        plt.savefig('../plots/ts/'+fn+'.png', bbox_inches='tight')
        plt.close()

    def plot_ts_spect(self):
        """
        It plots the time series with spectral components.
        """
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13., 8.), facecolor='w')
        df = self.df_obs
        # df[('dfo', 'S')]
        # self.df_obs['ORG'].plot(ax=ax[0,0], c='k', title='ORG') # sharex=ax[1,0],
        for i,j,sp,t in [(0,0,'ORG', '(a)'), (0,1,'S','(b)'), (1,0,'M','(c)'), (1,1,'L','(d)')]:
            #df.loc[df.columns.get_level_values(1) == sp]['2020-06':'2020-08'].plot(ax=ax[i, j], title=t, legend=False)
            df[('dfo',sp)]['2020-06':'2020-08'].plot(ax=ax[i,j], title=t,legend=False, c='k')
            df[('dfr', sp)]['2020-06':'2020-08'].plot(ax=ax[i, j], title=t, legend=False, c='red')
            df[('dfa', sp)]['2020-06':'2020-08'].plot(ax=ax[i, j], title=t, legend=False, c='dodgerblue')
        # plt.subplots_adjust()
        fig.tight_layout()
        plt.savefig('spec.jpg') # bbox_inches='tight')
        plt.close()
        exit()

    def calc_mse(self, df1=None, df2=None):
        r = df1.corr(df2)
        print([((df2 - df1)**2).mean(),
                r,
                (df2.std() - (r * df1.std())) ** 2,
                (df1.std() ** 2) * (1. - r ** 2),])
        return [((df2 - df1)**2).mean(),
                (df2.std() - (r * df1.std())) ** 2,
                (df1.std() ** 2) * (1. - r ** 2),
                r]

    def calc_stats(self, df1=None, df2=None):
        return [df1.corr(df2),
                self.calc_cov(df1=df1, df2=df2),
                self.calc_var(df=df1), self.calc_var(df=df2),
                ]

    def plot_box_scatter(self, vals=None):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(13,8), sharex=True)
        ax[0].boxplot([[i[0] for i in vals['1']], [i[0] for i in vals['2']],
                       [i[0] for i in vals['3']], [i[0] for i in vals['4']]], showmeans=True, showfliers=True,
                       medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                       meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        ax[0].set_yscale('log')
        ax[0].set_ylabel('mse',fontsize=14)
        ax[0].set_title('(a)')
        ax[1].boxplot([[i[1] for i in vals['1']], [i[1] for i in vals['2']],
                       [i[1] for i in vals['3']], [i[1] for i in vals['4']]], showmeans=True, showfliers=True,
                      medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                      meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        ax[1].set_yscale('log')
        ax[1].set_ylabel('e2',fontsize=14)
        ax[1].set_title('(b)')
        ax[2].boxplot([[i[2] for i in vals['1']], [i[2] for i in vals['2']],
                       [i[2] for i in vals['3']], [i[2] for i in vals['4']]], showmeans=True, showfliers=True,
                      medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                      meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        ax[2].set_yscale('log')
        ax[2].set_ylabel('e3',fontsize=14)
        ax[2].set_title('(c)')
        ax[2].set_xticks(np.arange(1, 5, 1))
        ax[2].set_xticklabels(['dfr_s', 'dfa_s', 'dfr_m', 'dfa_m'], color='k')# rotation=90, ha='right')
        plt.savefig('mse_137.jpg', bbox_inches='tight')
        plt.close()

    def plot_box_scatter_stats(self, vals=None):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(13,8))
        ax[0].boxplot([[i[0] for i in vals['1']], [i[0] for i in vals['2']],
                       [i[0] for i in vals['3']], [i[0] for i in vals['4']]], showmeans=True, showfliers=True,
                       medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                       meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        # ax[0].set_yscale('log')
        ax[0].set_ylabel('r',fontsize=14)
        ax[0].set_ylim(0, 1)
        ax[0].set_title('(a)')
        ax[1].boxplot([[i[1] for i in vals['1']], [i[1] for i in vals['2']],
                       [i[1] for i in vals['3']], [i[1] for i in vals['4']]], showmeans=True, showfliers=True,
                      medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                      meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        ax[1].set_yscale('log')
        ax[1].set_ylabel('cov',fontsize=14)
        ax[1].set_title('(b)')
        # o m1 m2
        ax[2].boxplot([[i[2] for i in vals['1']], [i[3] for i in vals['1']], [i[3] for i in vals['2']],
                       [i[2] for i in vals['3']], [i[3] for i in vals['3']], [i[3] for i in vals['4']]],
                      showmeans=True, showfliers=True,
                      medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                      meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        ax[2].set_yscale('log')
        ax[2].set_ylabel('var',fontsize=14)
        ax[2].set_title('(c)')
        ax[0].set_xticks(np.arange(1, 5, 1))
        ax[0].set_xticklabels(['dfr_s', 'dfa_s', 'dfr_m', 'dfa_m'], color='k')
        ax[1].set_xticks(np.arange(1, 5, 1))
        ax[1].set_xticklabels(['dfr_s', 'dfa_s', 'dfr_m', 'dfa_m'], color='k')
        ax[2].set_xticks(np.arange(1, 7, 1))
        ax[2].set_xticklabels(['dfo_s', 'dfr_s', 'dfa_s', 'dfo_m', 'dfr_m', 'dfa_m'], color='k')  # rotation=90, ha='right')
        fig.tight_layout()
        # plt.show()
        plt.savefig('stats_137.jpg', bbox_inches='tight')
        plt.close()

    def calc_rel_var(self, df=None, name=None):
        vars=[]
        vt = self.calc_var(df=df[(name, 'ORG')])
        for sp in ['S','M','L']:
            vars.append((self.calc_var(df=df[(name, sp)])/vt)*100.)
        return vars

    def plot_box_scatter_vars(self, vals=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
        plt.boxplot([[i[0] for i in vals], [i[1] for i in vals],
                       [i[2] for i in vals]], showmeans=True, showfliers=True,
                       medianprops=dict(linestyle='-', linewidth=1.5, color='orange'),
                       meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        # ax[0].set_yscale('log')
        plt.ylabel('(%)',fontsize=14)
        plt.ylim(0, 100)
        plt.show()
        # ax[0].set_title('(a)')

    def run(self):
        """
        It reads the observation data and analyses them.
        """
        dfs = self.read_obs_data()
        res = {'1': [], '2': [], '3': [], '4': []}
        for sta in dfs.columns.levels[0]:
            # sta='8'
            df = dfs[sta]
            # to plot spect.png
            # self.df_obs = dfs[sta]
            # self.plot_ts_spect()
            # to plot given time series
            # df[(sta, 'dfo', 'ORG')]['2020-06'].plot(c='k')
            # plt.ylabel('O$_{3}$ (nmol mol$^{-1}$)')
            # fig.tight_layout()
            # plt.savefig('4ml.jpg')# bbox_inches='tight')
            # plt.close()
            res['1'].append(self.calc_mse(df1=df[('dfo', 'S')], df2=df[('dfr', 'S')]))
            res['2'].append(self.calc_mse(df1=df[('dfo', 'S')], df2=df[('dfa', 'S')]))
            res['3'].append(self.calc_mse(df1=df[('dfo', 'M')], df2=df[('dfr', 'M')]))
            res['4'].append(self.calc_mse(df1=df[('dfo', 'M')], df2=df[('dfa', 'M')]))
            #
            # res['1'].append(self.calc_stats(df1=df[('dfo', 'S')], df2=df[('dfr', 'S')]))
            # res['2'].append(self.calc_stats(df1=df[('dfo', 'S')], df2=df[('dfa', 'S')]))
            # res['3'].append(self.calc_stats(df1=df[('dfo', 'M')], df2=df[('dfr', 'M')]))
            # res['4'].append(self.calc_stats(df1=df[('dfo', 'M')], df2=df[('dfa', 'M')]))
        # plt.boxplot([res['1'], res['2'], res['3'], res['4']])
        self.plot_box_scatter_stats(vals=res)
        exit()

if __name__ == '__main__':
    Assess().run()