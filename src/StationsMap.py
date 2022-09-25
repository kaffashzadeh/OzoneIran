"""
This program is written to map the stations locations.
"""
# imports python standard libraries
import os
import sys
import inspect
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

# import local libraries
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir + '/../../../')
os.sys.path.insert(0, parentdir)

font = {'family':'sans Serif', 'weight':'bold', 'size':10}
matplotlib.rc('font', **font)

__author__='Najmeh Kaffashzadeh'
__author_email__ = 'najmeh.kaffashzadeh@gmail.com'


class Map:

    name = 'Map station location'

    def __init__(self):
        """
        This initializes the variables.
        """
        self.location = 'Iran'

    def read_meta(self):
        """
        It reads the meta data of the stations.
        """
        self.meta = pd.read_excel(sys.path[0] + '/Data/PollutantTabTehStationsList.xlsx',
                                  index_col=0, header=0)

    def plot_city_map(self, city=None):
        """
        It plots map of a city.

        Args:
            city(str): name of the city
        """
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        if city =='Tehran':
            map = Basemap(projection='merc', llcrnrlat=35.55, urcrnrlat=35.85,
                          llcrnrlon=51.2, urcrnrlon=51.6, lat_0=35, lon_0=51, resolution='h', ax=ax)
        elif city =='Tabriz':
            map = Basemap(projection='merc', llcrnrlat=38., urcrnrlat=38.2,
                          llcrnrlon=46.2, urcrnrlon=46.4, lat_0=38.1, lon_0=46.3, resolution='h', ax=ax)
        map.drawmapboundary()
        map.readshapefile('../data/iran-roads/iran_roads',
                          'iran_roads', drawbounds=True, color='grey')
        for sta in self.meta.index:
            x, y = map(self.meta.loc[sta][1], self.meta.loc[sta][0])
            plt.plot(x, y, markersize=10, marker='o', color='orange')
            plt.text(x, y, str(int(self.meta.loc[sta][3])), color='k')
        # fig.tight_layout()
        # plt.show()
        plt.savefig(city+'.jpg') # dpi=1000)
        plt.close()

    def plot_map(self):
        """
        It plots map of the Iran, with two boxes referring the cities..
        """
        # self.plot_city_map(city='Tehran')
        # self.plot_city_map(city='Tabriz')
        map = Basemap(projection='lcc', llcrnrlat=32, urcrnrlat=40,
                      llcrnrlon=44.5, urcrnrlon=62, lat_0=35, lon_0=50, resolution='h')
        map.shadedrelief()
        map.drawcountries()
        xs = [50.2, 53, 53, 50.2, 50.2]
        ys = [35, 35, 36, 36, 35]
        map.plot(xs, ys, latlon=True, c='purple')
        x, y = map(50.5, 35.4)
        plt.text(x, y, 'Tehran', color='k')
        xs = [45.5, 47.5, 47.5, 45.5, 45.5]
        ys = [37.5, 37.5, 38.8, 38.8, 37.5]
        map.plot(xs, ys, latlon=True, c='purple')
        x, y = map(45.7, 38)
        plt.text(x, y, 'Tabriz', color='k')
        # plt.show()
        plt.savefig('stations_map.jpg', bbox_inches='tight')
        plt.close()

    def run(self):
        """
        It creates the map of the stations location.
        """
        self.read_meta()
        self.plot_map()


if __name__ == '__main__':
    Map().run()

