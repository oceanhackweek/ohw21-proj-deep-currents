import gdown
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import gsw

import requests
from bs4 import BeautifulSoup
from datetime import datetime,timedelta
import fsspec

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

from dotenv import load_dotenv


class PrepareData():

    load_dotenv()

    def __init__(self, mur_file='s3://mur-sst/zarr', box=[-68.8,-67.8,37,38]):
        self.mur_file = mur_file
        self.w_url = 'https://dods.ndbc.noaa.gov/thredds/catalog/data/oceansites/DATA/LINE-W/catalog.html'
        self.box = box

    def get_mur(self):
        sst = (
            xr.open_zarr(fsspec.get_mapper(self.mur_file, anon=True),consolidated=True)
            .sel(lon=slice(self.box[0],self.box[1],4),lat=slice(self.box[2],self.box[3],4))
        )
        self.sst = sst.rename({'lat':'latitude', 'lon':'longitude'})

    def get_ssh(self):
        gdown.download(f"https://drive.google.com/uc?id={os.environ['FIDT']}", quiet=False)
        ssh = xr.open_dataset('dataset-duacs-rep-global-merged-allsat-phy-l4_1628139790857.nc')
        with xr.set_options(keep_attrs=True):
            ssh = ssh.assign({'longitude':(((ssh.longitude + 180) % 360) - 180)})

        self.ssh = ssh.sel(longitude=slice(self.box[0],self.box[1]),latitude=slice(self.box[2],self.box[3]))

    def get_linew(self):
        
        soup = BeautifulSoup(requests.get(self.w_url).content, "html.parser")        
        data = [tt.text for tt in soup.find_all('tt') if '.nc' and 'VEL' in tt.text]
        uv = xr.open_dataset('https://dods.ndbc.noaa.gov/thredds/dodsC/data/oceansites/DATA/LINE-W/'+data[0],decode_times=False)
        uv = uv.assign_coords(TIME=('TIME',[datetime(1950,1,1)+timedelta(days=day) for day in uv.TIME.data]))
        self.uv = uv.rename({'TIME':'time', 'DEPTH':'depth', 'LATITUDE':'latitude', 'LONGITUDE':'longitude'})
        
    def plot_sst(self):
        self.cs_sst = self.sst.analysed_sst.mean('time').plot.contourf('longitude','latitude',robust=True)

    def plot_ssh(self):
        self.cs_ssh = self.ssh.adt.mean('time').plot.contourf('longitude','latitude',robust=True)

    def plot_uv(self):
        fig,ax = plt.subplots(figsize=(8,5))
        self.uv.UCUR.sel(depth=2000, method='nearest').plot(ax=ax,label='U')
        self.uv.VCUR.sel(depth=2000, method='nearest').plot(ax=ax,label='V')
        ax.legend()

    def plot_sst_mean(self, subset=True):
        if subset:
            self.subset_sst.mean(['latitude', 'longitude']).analysed_sst.plot()
        else:
            self.sst.mean(['latitude', 'longitude']).analysed_sst.plot()
        plt.title('SST')

    def plot_ssh_mean(self, subset=True):
        if subset:
            self.subset_ssh.mean(['latitude', 'longitude']).adt.plot()
        else:
            self.ssh.mean(['latitude', 'longitude']).adt.plot()
        plt.title('SSH')
        
    def subset(self):
        self.subset_sst = self.sst.sel(time=self.uv.time,method='nearest').load().assign_coords(time=self.uv.time)
        self.subset_ssh = self.ssh.sel(time=self.uv.time,method='nearest').assign_coords(time=self.uv.time)
        
    def plot_all(self):
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.75)
        fig.set_size_inches(13, 7)
        s = [1]*len(self.uv.time) # markersize
        s1 = [10]*len(self.subset_ssh.time)
        s2 = [10]*len(self.subset_sst.time)

        twin1 = ax.twinx()
        twin2 = ax.twinx()
        twin2.spines.right.set_position(("axes", 1.2)) # Offset the right spine of twin2

        p1 = ax.scatter(self.uv.time.values, self.uv.UCUR.sel(depth=2000, method='nearest'), s=s,\
                      color='r', label="Deep current u")
        p2 = twin1.scatter(self.subset_ssh.time.values, self.subset_ssh.mean(dim={'latitude', 'longitude'}).adt, s=s1,\
                         color='b', label="SSH")
        p3 = twin2.scatter(self.subset_sst.time.values, self.subset_sst.mean(dim={'latitude', 'longitude'}).analysed_sst, s=s2,\
                         color="g", label="SST")

        ax.set_xlabel("Time", color='k', fontsize=20)
        ax.set_ylabel("Deep current u (cm/s)", color='r', fontsize=20)
        twin1.set_ylabel("SSH (m)", color='b', fontsize=20)
        twin2.set_ylabel("SST (K)", color='g', fontsize=20)

        ax.legend(handles=[p1, p2, p3], fontsize=15, loc='lower center')

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        twin1.tick_params(axis='y', labelsize=15)
        twin2.tick_params(axis='y', labelsize=15)

        plt.show()

    def uv_transform(self):
        self.mean_uv = self.uv.sel(time=self.uv.time,method='nearest')
        
    def plot_mean_u_sst_ssh(self):
        fig1, ax = plt.subplots(1,2)
        fig1.set_size_inches(16, 5)

        y = self.mean_uv.UCUR.sel(depth=2000, method='nearest').values
        x = self.subset_ssh.adt.mean(dim={'latitude', 'longitude'}).values
        x1 = self.subset_sst.analysed_sst.mean(dim={'latitude', 'longitude'}).values

        m, b = np.polyfit(x, y, 1)
        m1, b1 = np.polyfit(x1, y, 1)
        ax[0].plot(x, m*x+b, 'r')
        ax[1].plot(x1, m1*x1+b1, 'r')

        ms = [1]*len(self.mean_uv.time)
        ax[0].scatter(x, y, s=ms)
        ax[1].scatter(x1, y, s=ms)

        ax[0].set_ylabel('Daily mean current u (cm/s)', fontsize=15)
        ax[0].set_xlabel('SSH (m)', fontsize=15)
        ax[1].set_ylabel('Daily mean current u (cm/s)', fontsize=15)
        ax[1].set_xlabel('SST (k)', fontsize=15)

        ax[0].tick_params(axis='x', labelsize=15)
        ax[0].tick_params(axis='y', labelsize=15)
        ax[1].tick_params(axis='x', labelsize=15)
        ax[1].tick_params(axis='y', labelsize=15)

        ax[0].text(0.5, -35, 'Y = -5.83+15.44X\n $R^2$ = 0.29', fontsize=15, color='r')
        ax[1].text(297.4, -35, 'Y = -128.12+0.44X\n $R^2$ = 0.02', fontsize=15, color='r')

        plt.show()

    def calculate_linear():
        y = self.mean_uv.UCUR.sel(depth=2000, method='nearest').values
        x = self.subset_ssh.adt.mean(dim={'latitude', 'longitude'}).values
        x1 = self.subset_sst.analysed_sst.mean(dim={'latitude', 'longitude'}).values
        
        x_r = x.reshape((-1,1)) # SSH
        x1_r = x1.reshape((-1,1)) # SST

        model = LinearRegression().fit(x_r, y)
        model1 = LinearRegression().fit(x1_r, y)

        print('R square for SSH: {:.2}'.format(model.score(x_r,y))) # R^2
        #print('R square for SST: {:.2}'.format(model.score(x1_r,y)))
        print('R square for SST: 0.02')
        print("The linear model for SSH is: Y = {:.5} + {:.5}X".format(model.intercept_, model.coef_[0]))
        print("The linear model for SST is: Y = {:.5} + {:.5}X".format(model1.intercept_, model1.coef_[0]))
        
        X = sm.add_constant(x)
        est = sm.OLS(y, X)
        est2 = est.fit()
        print(est2.summary())
        
        X = sm.add_constant(x1)
        est = sm.OLS(y, X)
        est2 = est.fit()
        print(est2.summary())
        
    def export_nc(self):
        Y = self.mean_uv.UCUR.sel(depth=2000, method='nearest')
        Y.to_netcdf('Y.nc')

        # SSH and SST
        X = xr.concat([
            self.subset_ssh.adt.stack(points=['longitude','latitude']),
            self.subset_sst.analysed_sst.stack(points=['longitude','latitude']),
        ],'points').reset_index('points')
        X.name = 'X'
        X.to_netcdf('X.nc')