import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


directory = '../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/'

all_file_list = os.listdir(directory)
file_list = []
string1 = 'baseline_gamma'

mean_hist = []
std_mean_hist = []
std_hist = []
baseline = []

for fi in all_file_list:
    if string1 in fi:
        print(fi)
        file_list.append(directory + fi)

        data = np.loadtxt(directory + fi)
        data = data[~np.isnan(data[:,0]),:]


        mean_hist.append(np.mean(data[:,0]))
        std_mean_hist.append(np.std(data[:,0]))
        std_hist.append(np.mean(data[:,1]))

        baseline.append(int(fi[28:30])) # beginning 26:28, end 28:30

fig = plt.figure(figsize=(10,8))
plt.plot(baseline,mean_hist,'.')
plt.xlabel('baseline1')
plt.ylabel('mean baseline')

fig = plt.figure(figsize=(10,8))
plt.plot(baseline,std_mean_hist,'.')
plt.xlabel('baseline1')
plt.ylabel('std mean baseline')

fig = plt.figure(figsize=(10,8))
plt.plot(baseline,std_hist,'.')
plt.xlabel('baseline1')
plt.ylabel('std baseline')

plt.show()



"""
directory0 = '../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/baseline_tests/baseline_beginning/'
directory1 = '../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/baseline_tests/baseline_end/'

file_safe0 = 'baseline_gamma_ze0_az0_bas0901.txt'
file_safe1 = 'baseline_gamma_ze0_az0_bas0115.txt'

data_safe0 = np.loadtxt(directory0 + file_safe0)
data_safe1 = np.loadtxt(directory1 + file_safe1)


all_file_list = os.listdir(directory1)
file_list = []
string1 = 'baseline_gamma'

diff_base = []
baseline = []
std_diff_base = []
for fi in all_file_list:
    
    if string1 in fi:
        
        print(fi)
        
        data = np.loadtxt(directory1 + fi)
        #print(len(data))
        #print(len(data_safe0))
        data_safe0e = data_safe0[~np.isnan(data[:,0]),:]
        data = data[~np.isnan(data[:,0]),:]
        
        diff_base.append(np.mean(data[:,0]-data_safe0e[:,0]))
        std_diff_base.append(np.std(data[:,0]-data_safe0e[:,0]))
        baseline.append(int(fi[28:30])) # beginning 26:28, end 28:30
  
fig = plt.figure(figsize=(10,8))
plt.errorbar(baseline,diff_base,yerr=std_diff_base,fmt='.')
plt.show()

"""
        
