#%%
# import
import numpy as np
import scipy
from scipy.stats import norm
from scipy.stats import entropy
from scipy import integrate
import matplotlib.pyplot as plt
plt.style.use("default")
#%%
# 正規分布を定義(scipy.stats.normだと積分が実行できない・・)
def norm_f(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x-mu,2)/(2*np.power(sigma,2)))
#%%
# 正規分布の相互情報量を定義（非積分関数）
def mutual_information(x, p_mu, p_sigma, q_mu, q_sigma):
    if norm_f(x,q_mu,q_sigma)==0.0 or norm_f(x,p_mu,p_sigma)==0.0:
        return 0 #発散回避
    else:
        return norm_f(x,q_mu,q_sigma)*(scipy.log(norm_f(x,q_mu,q_sigma))-scipy.log(norm_f(x,p_mu,p_sigma)))
#%%
# KL-divergenceの計算
def calc_kl(p_mu, p_sigma, q_mu, q_sigma):
    kl=integrate.quad(mutual_information,-np.inf,np.inf,args=(p_mu,p_sigma,q_mu,q_sigma))
    return kl
#%%
# q_mu,q_sigmaを(0,1)に固定し、p_mu=0の状況下でp_sigmaを変えた時の挙動をみる
q_mu = 0
q_sigma = 1
p_mu = 0

p_sigma_arr=np.arange(0.1,5,0.1)
kl_arr=[calc_kl(p_mu,p_sigma,q_mu,q_sigma)[0] for p_sigma in p_sigma_arr]
plt.plot(p_sigma_arr,kl_arr)
plt.xticks(ticks=np.arange(0,5,1))
plt.ylim(0,1)
plt.savefig("p_sig_change.png")
#%%
# p_mu,p_sigmaを(0,1)に固定し、q_mu=0の状況下でq_sigmaを変えた時の挙動をみる
q_mu = 0
p_mu = 0
p_sigma = 1

q_sigma_arr=np.arange(0.1,5,0.1)
kl_arr=[calc_kl(p_mu,p_sigma,q_mu,q_sigma)[0] for q_sigma in q_sigma_arr]
plt.plot(q_sigma_arr,kl_arr)
plt.xticks(ticks=np.arange(0,5,1))
plt.ylim(0,1)
#%%
