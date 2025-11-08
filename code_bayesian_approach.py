
"""
May 3rd 2025

@author: Zoe Lyn Busche

project: Bayesian Analysis: A Study of Educational Returns to Income in Germany 

- Durbin Wu Hausman test for endogeneity
- Instrument robustness test with different gamma_tilde
- Comparing effect of education on income based on 
        - Gender
        - West or east Germany
        - Enterpreneur or employed
        - Level of education 

"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import wishart, multivariate_normal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm



def durbin_wu_hausman(df):
    '''
    H_0 = explanatory variable (education) is exogenous
    H_1 = explanatory variable is endogenous
    
    if p-value of residual is less than 0.05, reject H_0 --> endogeneity
    '''
    
    # first-stage regression (education and education_father)
    first_stage = sm.OLS(df['education'], sm.add_constant(df[['education_father']])).fit() 
    df['residual'] = df['education'] - first_stage.fittedvalues  # residuals

    # second-stage regression (log_income and education + residual)
    second_stage = sm.OLS(df['log_income'], sm.add_constant(df[['education', 'residual']])).fit()

    print(second_stage.summary())


def corr_map(df):
    '''
    generates heatmap to see correlation between selected variables
    prints correlation matrix with corresponding p-values
    '''
    #select columns of interest for the heatmap
    df_heat = df[['log_income', 'education', 'education_father', 'experience', 'experience_squared', 'unemployment', 'wealth']] 
    corr_matrix = df_heat.corr()
    heatmap = sns.heatmap(df_heat.corr(), cmap="Reds",
                          vmax=.8, square=True, annot=True, fmt='.2f')

    #compute p-values
    pval_matrix = pd.DataFrame(np.ones(
        corr_matrix.shape), columns=corr_matrix.columns, index=corr_matrix.index)
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr, pval = pearsonr(df_heat[col1], df_heat[col2])
            pval_matrix.loc[col1, col2] = pval
            pval_matrix.loc[col2, col1] = pval  # symmetric

    print(heatmap)
    print('\n###### correlation matrix #####')
    print(corr_matrix)
    print('\n#### corresponding p-values')
    print(pval_matrix)


def cal_posterior(y, x, z, W, n_rows, gamma_tilde):

    #instruments and matrices
    Ztilde = np.hstack((z.reshape(-1, 1), W))
    num_controls = W.shape[1]
    num_instr = Ztilde.shape[1]

    #initial values
    delta2_init = inv(Ztilde.T @ Ztilde) @ (Ztilde.T @ x)       #x on Ztilde
    x_hat = Ztilde @ delta2_init                                #fitted x
    
    Xhat_W = np.hstack((x_hat.reshape(-1, 1), W))
    beta_delta1_init = inv(Xhat_W.T @ Xhat_W) @ (Xhat_W.T @ y)  #y on x_hat + W
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

    #storage
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
        cov_mat_2 = (cov_mat_2 + cov_mat_2.T) / 2

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
    
    #uncomment this for posterior density plot
    '''
    #plot density of posterior distribution of beta
    beta_series = pd.Series(beta_post)
    beta_series.plot.density(bw_method='scott', color='deeppink', linestyle='-', linewidth=2)
    plt.xlabel('β', fontsize = 11)
    plt.ylabel('p(β|data)', fontsize = 11)
    plt.show()
    '''
    return (np.mean(beta_post), np.std(beta_post), np.percentile(beta_post, [2.5, 97.5]), df_control_stats)



def cal_diff_gamma_tilde(y, x, z, W, n_rows):
    '''
    returns the posterior mean, std deviation and percentiles of beta for different gamma_tilde
    gamma_tilde in [-0,5, 100]
    '''

    #save values for mean and percentiles of beta distribution
    mean_beta = []
    percentile_low = []
    percentile_up = []
    gamma_tilde_values = []

    #go through values -0.5 until 1 for gamma_tilde
    gamma_tilde = -0.5
    

    while gamma_tilde <= 1:
        
        mean_beta_post, std_dev_beta_post, percentiles_beta_post, control_stats = cal_posterior(
            y, x, z, W, n_rows, gamma_tilde)
        print(f"Posterior mean of β with gamma_tilde = {gamma_tilde:.1f}: {mean_beta_post:.4f}")
        print(f"std deviation of β with gamma_tilde = {gamma_tilde:.1f}: {std_dev_beta_post:.4f}")
        print(f"percentiles of β with gamma_tilde = {gamma_tilde:.1f}: {percentiles_beta_post}\n")

        mean_beta.append(mean_beta_post)
        percentile_low.append(percentiles_beta_post[0])
        percentile_up.append(percentiles_beta_post[1])
        gamma_tilde_values.append(gamma_tilde)

        gamma_tilde += 0.1
        
    
    #gamma_tilde equal to 5, 10, 50 and 100

    #gamma_tilde = 5
    mean_beta_post, std_dev_beta_post, percentiles_beta_post, control_stats = cal_posterior(
        y, x, z, W, n_rows, 5)
    print(f"Posterior mean of β with gamma_tilde = 5: {mean_beta_post:.4f}")
    print(f"std deviation of β with gamma_tilde = 5: {std_dev_beta_post:.4f}")
    print(f"percentiles of β with gamma_tilde = 5: {percentiles_beta_post}\n")
    
    #gamma_tilde = 10
    mean_beta_post, std_dev_beta_post, percentiles_beta_post, control_stats = cal_posterior(
       y, x, z, W, n_rows, 10)
    print(f"Posterior mean of β with gamma_tilde = 10: {mean_beta_post:.4f}")
    print(f"std deviation of β with gamma_tilde = 10: {std_dev_beta_post:.4f}")
    print(f"percentiles of β with gamma_tilde = 10: {percentiles_beta_post}\n")

    #gamma_tilde = 50
    mean_beta_post, std_dev_beta_post, percentiles_beta_post, control_stats = cal_posterior(
        y, x, z, W, n_rows, 50)
    print(f"Posterior mean of β with gamma_tilde = 50: {mean_beta_post:.4f}")
    print(f"std deviation of β with gamma_tilde = 50: {std_dev_beta_post:.4f}")
    print(f"percentiles of β with gamma_tilde = 50: {percentiles_beta_post}\n")
    
    #gamma_tilde = 100
    mean_beta_post, std_dev_beta_post, percentiles_beta_post, control_stats = cal_posterior(
        y, x, z, W, n_rows, 100)
    print(f"Posterior mean of β with gamma_tilde = 100: {mean_beta_post:.4f}")
    print(f"std deviation of β with gamma_tilde = 100: {std_dev_beta_post:.4f}")
    print(f"percentiles of β with gamma_tilde = 100: {percentiles_beta_post}\n")
    
    
    #plot graph with mean and percentiles for different gamma_tilde from -0.5 to 1
    plt.plot(gamma_tilde_values, mean_beta,
             label='posterior mean', color='deeppink')
    plt.plot(gamma_tilde_values, percentile_low,
             label='percentiles', color='black', linestyle='dashed')
    plt.plot(gamma_tilde_values, percentile_up,
             color='black', linestyle='dashed')
    plt.xlabel('gamma_tilde')
    plt.ylabel('posterior mean and percentiles of β distribution')
    plt.legend()
    plt.rcParams['figure.dpi'] = 400
    plt.show()


#load dataset and define variables
data = pd.read_excel("German_data_2004.xls")
y = data.iloc[:, 0].values           #log(income)
x = data.iloc[:, 1].values           #education
z = data.iloc[:, 2].values           #father's education
n_rows, n_cols = data.shape
W = np.hstack((data.iloc[:, 4:].values, np.ones((n_rows, 1))))  #control 

#Durbin Wu Hausman test for endogeneity 
print('\n##### Durbin-Wu-Hausman test #####\n')
durbin_wu_hausman(data)

#correlation matrix with p-values and heatmap for variables of interest
corr_map(data)


########## GAMMA_TILDE = 0 ##########

mean_beta_post, std_dev_beta_post, percentiles_beta_post, control_stats = cal_posterior(y, x, z, W, n_rows, 0)
print('\n##### results for gamma_tilde=0 ######\n')
print(f"\nposterior mean of β: {mean_beta_post:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post}")
print(f"Posterior 95% CI for β: {percentiles_beta_post}\n")
print('\ncontrol variables:\n')
print(control_stats)


########## DIFFERENT GAMMA_TILDE ##########

print('##### different gamma_tilde #####\n')
cal_diff_gamma_tilde(y, x, z, W, n_rows)


########## GENDER ##########

#divide dataset into two datasets for female and male
df_female = data[data.iloc[:, 11] == 0]
df_female = df_female.drop('male', axis=1)
df_male = data[data.iloc[:, 11] == 1]
df_male = df_male.drop('male', axis=1)

#female (perfectly valid instrument)
print('\n##### female #####\n')
print('shape of dataset: ', df_female.shape)

y_female = df_female.iloc[:, 0].values              #log(income)
x_female = df_female.iloc[:, 1].values              #education for female
z_female = df_female.iloc[:, 2].values              #father's education
n_rows_female, n_cols_female = df_female.shape
W_female = np.hstack((df_female.iloc[:, 3:].values, np.ones((n_rows_female, 1))))  #control 
gamma_tilde = 0

#results for female
mean_beta_post_female, std_dev_beta_post_female, percentiles_beta_post_female, control_stats_female = cal_posterior(y_female, x_female, z_female, W_female, n_rows_female, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_female:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_female}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_female}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_female)

#male (perfectly valid instrument)
print('\n##### male #####\n')
print('shape of dataset: ', df_male.shape)

y_male = df_male.iloc[:, 0].values              #log(income)
x_male = df_male.iloc[:, 1].values              #education
z_male = df_male.iloc[:, 2].values              #father's education
n_rows_male, n_cols_male = df_male.shape
W_male = np.hstack((df_male.iloc[:, 3:].values, np.ones((n_rows_male, 1))))  #control 
gamma_tilde = 0

#results for male
mean_beta_post_male, std_dev_beta_post_male, percentiles_beta_post_male, control_stats_male = cal_posterior(y_male, x_male, z_male, W_male, n_rows_male, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_male:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_male}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_male}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_male)


########## WEST AND EAST GERMANY ##########

#divide dataset into two datasets for east and west germany
df_west = data[data.iloc[:, 10] == 1]
df_west = df_west.drop('west_germany', axis=1)
df_east = data[data.iloc[:, 10] == 0]
df_east = df_east.drop('west_germany', axis=1)

#west (perfectly valid instrument)
print('\n##### west #####\n')
print('shape of dataset: ', df_west.shape)

y_west = df_west.iloc[:, 0].values              #log(income)
x_west = df_west.iloc[:, 1].values              #education for west
z_west = df_west.iloc[:, 2].values              #father's education
n_rows_west, n_cols_west = df_west.shape
W_west = np.hstack((df_west.iloc[:, 3:].values, np.ones((n_rows_west, 1))))  #control 
gamma_tilde = 0

#results for west germany
mean_beta_post_west, std_dev_beta_post_west, percentiles_beta_post_west, control_stats_west = cal_posterior(y_west, x_west, z_west, W_west, n_rows_west, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_west:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_west}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_west}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_west)

#east (perfectly valid instrument)
print('\n##### east #####\n')
print('shape of dataset: ', df_east.shape)

y_east = df_east.iloc[:, 0].values              #log(income)
x_east = df_east.iloc[:, 1].values              #education for east
z_east = df_east.iloc[:, 2].values              #father's education
n_rows_east, n_cols_east = df_east.shape
W_east = np.hstack((df_east.iloc[:, 3:].values, np.ones((n_rows_east, 1))))  #control 
gamma_tilde = 0

#results for east germany
mean_beta_post_east, std_dev_beta_post_east, percentiles_beta_post_east, control_stats_east = cal_posterior(y_east, x_east, z_east, W_east, n_rows_east, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_east:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_east}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_east}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_east)


########## ENTREPRENEURS AND EMPLOYEES ##########

#divide dataset into two datasets for selfemployed people and entrepreneurs
df_selfemployed = data[data.iloc[:, 3] == 1]
df_employed = data[data.iloc[:, 3] == 0]  # select rows with employees

#enterpreneur (perfectly valid instrument)

print('##### selfemployed ####\n')
print('shape of dataset: ', df_selfemployed.shape)

y_self = df_selfemployed.iloc[:, 0].values           #log(income)
x_self = df_selfemployed.iloc[:, 1].values           #education
z_self = df_selfemployed.iloc[:, 2].values           #father's education
n_rows_self, n_cols_self = df_selfemployed.shape
W_self = np.hstack((df_selfemployed.iloc[:, 4:].values, np.ones((n_rows_self, 1))))  #control without 'dummy_selfemployed'
gamma_tilde = 0

#results for selfemployed
mean_beta_post_self, std_dev_beta_post_self, percentiles_beta_post_self, control_stats_self = cal_posterior(y_self, x_self, z_self, W_self, n_rows_self, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_self:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_self}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_self}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_self)

#employee
print('##### employed ####\n')
print('shape of dataset: ', df_employed.shape)

y_em = df_employed.iloc[:, 0].values        #log(income)
x_em = df_employed.iloc[:, 1].values        #education
z_em = df_employed.iloc[:, 2].values        #father's education
n_rows_em, n_cols_em = df_employed.shape
W_em = np.hstack((df_employed.iloc[:, 4:].values, np.ones((n_rows_em, 1))))  # control
gamma_tilde = 0

#results for employed
mean_beta_post_em, std_dev_beta_post_em, percentiles_beta_post_em, control_stats_em = cal_posterior(y_em, x_em, z_em, W_em, n_rows_em, gamma_tilde)
print(f"Posterior mean of β: {mean_beta_post_em:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_em}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_em}")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_em)


########## YEARS OF EDUCATION ##########

#check how many times unique values of years of education 
unique_values_education = data['education'].value_counts()
unique_values_education.columns = ['education', 'count']
print(unique_values_education)

#divide data into higher and lower levels of education
df_low = data[data['education'] < 13]
df_high = data[data['education'] >= 13]

#size of subsets
print('\nshape of subsets:') 
print(f'low education: {df_low.shape} \nhigher education: {df_high.shape}')    

#check if correct values for education for each subset
print('\nunique values in education for each subset:')
print('low:', df_low['education'].unique())
print('high:', df_high['education'].unique())

#High Education
y_h = df_high.iloc[:, 0].values           #log(income)
x_h = df_high.iloc[:, 1].values           #education
z_h = df_high.iloc[:, 2].values           #father's education
n_rows_h, n_cols_h = df_high.shape
W_h = np.hstack((df_high.iloc[:, 4:].values, np.ones((n_rows_h, 1))))  
gamma_tilde = 0

#results for high education
print('\n##### High Education #####')
mean_beta_post_h, std_dev_beta_post_h, percentiles_beta_post_h, control_stats_h = cal_posterior(y_h, x_h, z_h, W_h, n_rows_h, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_h:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_h}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_h}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_h)

#Low Education
y_l = df_low.iloc[:, 0].values           #log(income)
x_l = df_low.iloc[:, 1].values           #education
z_l = df_low.iloc[:, 2].values           #father's education
n_rows_l, n_cols_l = df_low.shape
W_l = np.hstack((df_low.iloc[:, 4:].values, np.ones((n_rows_l, 1))))  
gamma_tilde = 0

#results high education
print('\n##### Low Education #####')
mean_beta_post_l, std_dev_beta_post_l, percentiles_beta_post_l, control_stats_l = cal_posterior(y_l, x_l, z_l, W_l, n_rows_l, gamma_tilde)
print(f"\nposterior mean of β: {mean_beta_post_l:.4f}")
print(f"posterior standard deviation for β: {std_dev_beta_post_l}")
print(f"Posterior 95% CI for β: {percentiles_beta_post_l}\n")

#mean, std deviation and percentiles of controls
print('\nPosterior results for control variables:\n')
print(control_stats_l)

