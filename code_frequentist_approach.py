
"""
May 23rd 2025

@author: Zoe Lyn Busche

project: Frequentist Analysis: A Study of Educational Returns to Income in Germany 

- 2-Stage-Least-Square (2SLS) and Ordenary Least Square (OLS) regression
- Density plots comparing Bayesian IV, 2SLS, and OLS
- Comparing effect of education on income based on 
        - Gender
        - West or east Germany
        - Enterpreneur or employed
        - Level of education

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import wishart, multivariate_normal

def TSLS_regression(data, control_split):
    '''
    2 stage least squares regression on education using father's education as instrument
    '''
    
    y = data['log_income']
    x = data['education']
    W = data.columns[control_split:].tolist()
    z = data[['education_father'] + W]
    z = sm.add_constant(z)
    
    #first stage: regress education on instrument and controls
    first_stage = sm.OLS(x, z).fit()
    data['education_hat'] = first_stage.fittedvalues
    
    #second stage: segress log_income on pred education and controls
    x_hat = data[['education_hat'] + W]
    x_hat = sm.add_constant(x_hat)
    second_stage = sm.OLS(y, x_hat).fit()
    
    print(second_stage.summary())
    return second_stage


def OLS_regression(data):
    y = data['log_income']
    W = data.columns[3:].tolist()
    x = data[['education'] + W]
    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    print(model.summary())
    return model


def cal_beta_post(y, x, z, W, n_rows, gamma_tilde):

    #instruments and matrices
    Ztilde = np.hstack((z.reshape(-1, 1), W))
    num_controls = W.shape[1]
    num_instr = Ztilde.shape[1]
    
    delta2_init = inv(Ztilde.T @ Ztilde) @ (Ztilde.T @ x)
    x_hat = Ztilde @ delta2_init
    Xhat_W = np.hstack((x_hat.reshape(-1, 1), W))
    beta_delta1_init = inv(Xhat_W.T @ Xhat_W) @ (Xhat_W.T @ y)
    beta = beta_delta1_init[0]
    delta1 = beta_delta1_init[1:]
    
    u1 = y - Xhat_W @ beta_delta1_init
    u2 = x - Ztilde @ delta2_init
    Sigma_u = np.cov(np.vstack((u1, u2)))
    Sigma_u_inv = inv(Sigma_u)
    delta2 = delta2_init.copy()
    
    #prior
    beta_prior_mean = 0.0
    beta_prior_var = 1.0
    beta_prior_var_inv = 1 / beta_prior_var
    
    #gibbs parameters
    n_draws = 11000
    burn_in = 1000
    #gamma_tilde = 0.0  # for strict exclusion restriction
    
    #store values
    beta_samples = np.zeros(n_draws)
    delta1_samples = np.zeros((n_draws, num_controls))
    delta2_samples = np.zeros((n_draws, num_instr))
    sigma_u_samples = np.zeros((n_draws, 4))
    Ztilde_T_Ztilde_inv = inv(Ztilde.T @ Ztilde)
    
    #gibbs sampling
    for i in range(n_draws):
    
        #sample Sigma_u
        u1 = y - ((x + z * gamma_tilde) * beta) - W @ delta1
        u2 = x - Ztilde @ delta2
        U = np.column_stack((u1, u2))
        S = inv(U.T @ U)
        Sigma_u_inv = wishart.rvs(df=n_rows, scale=S)
        Sigma_u = inv(Sigma_u_inv)
    
        #sample beta and delta1
        X_star = np.column_stack(((x + z * gamma_tilde), W))
        cond_mean_u1 = (x - Ztilde @ delta2) * Sigma_u[0, 1] / Sigma_u[1, 1]
        cond_var_u1 = Sigma_u[0, 0] - (Sigma_u[0, 1] ** 2) / Sigma_u[1, 1]
    
        cov_inv = np.eye(1 + num_controls) * 0
        cov_inv[0, 0] = beta_prior_var_inv
        cov_inv += (1 / cond_var_u1) * (X_star.T @ X_star)
    
        cov_mat = inv(cov_inv)
        cov_mat = (cov_mat + cov_mat.T) / 2
        mean_vec = cov_mat @ ((beta_prior_var_inv * beta_prior_mean) +
                              (1 / cond_var_u1) * X_star.T @ (y - cond_mean_u1))
    
        beta_delta1 = multivariate_normal.rvs(mean=mean_vec, cov=cov_mat)
        beta = beta_delta1[0]
        delta1 = beta_delta1[1:]
    
        #sample delta2
        cond_mean_u2 = (y - ((x + z * gamma_tilde) * beta) - W @
                        delta1) * Sigma_u[0, 1] / Sigma_u[0, 0]
        cond_var_u2 = Sigma_u[1, 1] - (Sigma_u[0, 1] ** 2) / Sigma_u[0, 0]
    
        cov_mat_2 = cond_var_u2 * Ztilde_T_Ztilde_inv
        cov_mat_2 = (cov_mat_2 + cov_mat_2.T) / 2  # enforce symmetry
    
        mean_vec_2 = (1 / cond_var_u2) * Ztilde.T @ (x - cond_mean_u2)
    
        delta2 = multivariate_normal.rvs(
            mean=cov_mat_2 @ mean_vec_2, cov=cov_mat_2)
    
        #store draws
        beta_samples[i] = beta
        delta1_samples[i] = delta1
        delta2_samples[i] = delta2
        sigma_u_samples[i] = [Sigma_u[0, 0],
                              Sigma_u[0, 1], Sigma_u[1, 0], Sigma_u[1, 1]]
    
    #post-processing
    beta_post = beta_samples[burn_in:]
    delta1_post = delta1_samples[burn_in:]
    delta2_post = delta2_samples[burn_in:]
    sigma_u_post = sigma_u_samples[burn_in:]
    
    #compute statistics for controls
    control_stats = np.zeros((num_controls, 4))
    for j in range(num_controls):
        draws = delta1_post[:, j]
        control_stats[j, 0] = np.mean(draws)
        control_stats[j, 1] = np.std(draws)
        control_stats[j, 2] = np.percentile(draws, 2.5)
        control_stats[j, 3] = np.percentile(draws, 97.5)
        
    df_control_stats = pd.DataFrame(control_stats, columns=["mean", "std deviation", "2.5%", "97.5%"])
    df_control_stats.index = data.columns[3:]
    return beta_post



def compare_density_plot(model, second_stage, beta_post):

    #OLS
    beta_ols = model.params['education']
    se_ols = model.bse['education']
    
    #2SLS
    beta_2sls = second_stage.params['education_hat']
    se_2sls = second_stage.bse['education_hat']
    
    #create x-axis range
    lower = min(beta_post.min(), beta_ols - 4 * se_ols, beta_2sls - 4 * se_2sls)
    upper = max(beta_post.max(), beta_ols + 4 * se_ols, beta_2sls + 4 * se_2sls)
    beta_range = np.linspace(lower, upper, 1000)
    
    #compute normal densities for OLS and 2SLS (assume normal distribution)
    ols_density = (1 / (se_ols * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((beta_range - beta_ols) / se_ols)**2)
    sls_density = (1 / (se_2sls * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((beta_range - beta_2sls) / se_2sls)**2)
    
    #plot
    plt.figure(figsize=(8, 5))
    plt.plot(beta_range, ols_density, label='OLS', color='blueviolet', linestyle=':', linewidth=2)
    plt.plot(beta_range, sls_density, label='2SLS', color='purple', linestyle='--', linewidth=2)
    beta_series = pd.Series(beta_post)
    beta_series.plot.density(bw_method='scott', label='Bayesian IV', color='deeppink', linestyle='-.', linewidth=2)
    
    #labels and legend
    plt.xlabel(r'$\beta$ (effect of education)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.axvline(beta_ols, color='blueviolet', linestyle=':', linewidth=1)
    plt.axvline(beta_2sls, color='purple', linestyle=':', linewidth=1)
    plt.axvline(beta_series.mean(), color='deeppink', linestyle=':', linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


data = pd.read_excel("German_data_2004.xls")

########## RESULTS WHOLE DATA ##########

#2SLS regression
print('\n##### results 2SLS regression (whole dataset) #####\n')
control_split = 3
second_stage = TSLS_regression(data, control_split)
data = data.drop(columns=['education_hat'])

#OLS regression
print('\n##### OLS regression (whole dataset) #####\n')
model= OLS_regression(data)


########## DENSITY PLOT ##########

#calculate beta_post from Bayesian approach
data = pd.read_excel("German_data_2004.xls")
y = data.iloc[:, 0].values           #log(income)
x = data.iloc[:, 1].values           #education
z = data.iloc[:, 2].values           #father's education
n_rows, n_cols = data.shape
W = np.hstack((data.iloc[:, 4:].values, np.ones((n_rows, 1))))  #control 
gamma_tilde = 0

beta_post = cal_beta_post(y, x, z, W, n_rows, gamma_tilde)

#print density plot for OLS, 2SLS and Bayesian
compare_density_plot(model, second_stage, beta_post)


########## EDUCATION LEVEL ##########

#divide data into higher and lower levels of education
df_low = data[data['education'] < 13]
df_high = data[data['education'] >= 13]
control_split = 3

#higher education
print('\n##### higher education: results for 2SLS regression #####\n')
TSLS_regression(df_high,  control_split)
df_high = df_high.drop(columns=['education_hat'])

#lower education
print('\n##### lower education: results for 2SLS regression #####\n')
TSLS_regression(df_low,  control_split)
df_low = df_low.drop(columns=['education_hat'])


########## GENDER ##########

#divide dataset into two datasets for female and male
df_female = data[data.iloc[:, 11] == 0]
df_female = df_female.drop('male', axis=1)
df_male = data[data.iloc[:, 11] == 1]
df_male = df_male.drop('male', axis=1)
control_split = 3

#female
print('\n##### female: results for 2SLS regression #####\n')
TSLS_regression(df_female,  control_split)
df_female = df_female.drop(columns=['education_hat'])

#male
print('\n##### male: results for 2SLS regression #####\n')
TSLS_regression(df_male,  control_split)
df_male = df_male.drop(columns=['education_hat'])


########## WEST & EAST GERMANY ##########

#divide dataset into two different sets for west and east germany
df_east = data[data.iloc[:, 10] == 0]
df_east = df_east.drop('west_germany', axis=1)
df_west = data[data.iloc[:, 10] == 1]
df_west = df_west.drop('west_germany', axis=1)
control_split = 3

#east
print('\n##### east: results for 2SLS regression #####\n')
TSLS_regression(df_east,  control_split)
df_east = df_east.drop(columns=['education_hat'])
print(df_east.shape)

#west
print('\n##### west: results for 2SLS regression #####\n')
TSLS_regression(df_west,  control_split)
df_west = df_west.drop(columns=['education_hat'])
print(df_west.shape)


########## ENTREPRENEUR & EMPLOYEE ##########

#divide dataset into two datasets for employed people and entrepreneurs
df_selfemployed = data[data.iloc[:, 3] == 1]
df_employed = data[data.iloc[:, 3] == 0]  # select rows with employees
control_split = 4

#results for entrepreneurs
print('\n##### entrepreneurs: results for 2SLS regression #####\n')
TSLS_regression(df_selfemployed, control_split)
df_selfemployed = df_selfemployed.drop(columns=['education_hat'])
print(df_selfemployed.shape)

#results for selfemployed
print('\n##### employees: results for 2SLS regression  #####\n')
TSLS_regression(df_employed, control_split)
df_employed = df_employed.drop(columns=['education_hat'])
print(df_employed.shape)