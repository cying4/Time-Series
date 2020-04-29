import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal
#%%
#Check the dataset
df=pd.read_csv('AAPL.csv')
df.head(5)
#%%
def nan_checker(df):
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])
    df_nan = df_nan.sort_values(by='proportion', ascending=False)
    return df_nan
df_nan = nan_checker(df)
df_nan.reset_index(drop=True)
#No missing value
#%%
df.Timestamp = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df.Timestamp
df.drop('Date',axis = 1, inplace = True)
df.drop('Name',axis = 1, inplace = True)
df.head(5)
#%%
df.info()
#%%
plt.plot(df['High'])
plt.xlabel('date')
plt.ylabel('value')
plt.title('dependent variable versus time')
plt.show()
#%%
def get_auto_corr(timeSeries,k):
    l = len(timeSeries)
    timeSeries1 = timeSeries[0:l-k]
    timeSeries2 = timeSeries[k:]
    timeSeries_mean = np.mean(timeSeries)
    timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
    auto_corr = 0
    for i in range(l-k):
        temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
        auto_corr = auto_corr + temp
    return auto_corr
#%%
dep=np.array(df['High'])
acf=[]
for i in range(20):
    acf.append(get_auto_corr(dep,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for dependent variable')
plt.show()
#%%
from statsmodels.tsa.stattools import adfuller
stat =df['High'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#%%
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(df)
plt.show()
#%%
sns.heatmap(df)
plt.show()
#%%
df['first_high']=(df['High']-df['High'].shift(1)).dropna()
df=df.drop(df.index[0])
df.head()
#%%
stat =df['first_high'].values
result = adfuller(stat)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#%%
#get dependent and independent variable
df1=df[['first_high','Open','Volume']]
#df1['Volume']=df['Volume'].apply(lambda x: x/1000000)
#df1['Volume'].div(1000000)
df1.loc[:,'Volume']=df1.loc[:,'Volume'].div(1000000)
#df1['Volume'].round(2)
df2 = df1[df1['first_high'] >0]
df2.head()
#%%
df2.info()
#%%
from sklearn.model_selection import train_test_split
X=df2[['Open','Volume']]
Y=df2[['first_high']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,shuffle=False)
#%%
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df2['first_high'], model='additive', freq=1)
result.plot()
plt.title('Addtive seasonal')
plt.show()
#%%
result1 = seasonal_decompose(df2['first_high'], model='multiplicative',freq=1)
result1.plot()
plt.title('Multiplicative seasonal')
plt.show()
#%%
#Holt winter prediction
from statsmodels.tsa.api import ExponentialSmoothing
fit1 =ExponentialSmoothing(np.asarray(y_train['first_high']), seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)
y_test['Holt_Winter'] = fit1.forecast(len(y_test))
y_test.head()

#%%
ttt=np.array(y_test['first_high'])
error_winter=ttt-y_test['Holt_Winter'].values
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_winter,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values Holt Winter model error')
plt.show()
#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q_winter=len(error_winter)*np.sum(acf1**2)
var_winter=np.var(error_winter)
mse_winter=np.mean(error_winter**2)
mean_winter=np.mean(error_winter)
rmse_winter=(mse_winter)**0.5
print("The Q value of Holt Winter model is:",Q_winter)
print("The variance of Holt Winter model is:",var_winter)
print("The mse of Holt Winter model is:",mse_winter)
print("The mean of Holt Winter model error is:",mean_winter)
print("RMSE of Holt Winter error is:",rmse_winter)
#%%
#Regression model
x_train.insert(0,"int",1)
x_test.insert(0,"int",1)
#%%
model=sm.OLS(y_train,x_train).fit()
print(model.summary())
#%%
y_test['Regression'] = model.predict(x_test)
error_reg=y_test['first_high'].values-y_test['Regression'].values

#%%
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_reg,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values of regression model residual')
plt.show()

#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q_reg=len(error_reg)*np.sum(acf1**2)
var_reg=np.var(error_reg)
mse_reg=np.mean(error_reg**2)
mean_reg=np.mean(error_reg)
rmse_reg=(mse_reg)**0.5
print("The Q value of regression is:",Q_reg)
print("The variance of regression model error is:",np.var(error_reg))
print("The mse of regression model error is:",np.mean(error_reg**2))
print("The mean of regression model error is:",mean_reg)
print("RMSE of regression mdel error is:",rmse_reg)
#finish regression model

#%%
#ARMA model
#determin parameters
y=np.array(y_train['first_high'])
acf=[]
for i in range(100):
    acf.append(get_auto_corr(y,i+1))
ry=[np.var(y)]
for i in range(99):
    ry.append(acf[i+1]*np.var(y))
#%%
phi=[]
phi_1=[]
i=0
gpac = np.zeros(shape=(8, 7))
for j in range(0,8):
    for k in range(2,9):
        bottom = np.zeros(shape=(k, k))
        top = np.zeros(shape=(k, k))
        for m in range(k):
            for n in range(k):
                bottom[m][n]=ry[abs(j+m - n)]
            top[m][-1]=ry[abs(j+m+1)]
        i=i+1
        top[:,:k-1] = bottom[:,:k-1]
        phi.append(round((np.linalg.det(top) / np.linalg.det(bottom)),2))
    phi_1.append(round(ry[j + 1] / ry[j],2))
gpac=np.array(phi).reshape(8,7)
Phi1=pd.DataFrame(phi_1)
Gpac=pd.DataFrame(gpac)
GPAC = pd.concat([Phi1,Gpac], axis=1)
GPAC.columns=['1','2','3','4','5','6','7','8']
print(GPAC)
#%%
sns.heatmap(GPAC, center=0, annot=True)
plt.title("Generalized partial autocorrelation function ")
plt.xlabel("na")
plt.ylabel("nb")
plt.show()
#%%
#na=1,nb=1
model1=sm.tsa.ARMA(y,(1,1)).fit(trend='nc',disp=0)
print(model1.summary())
#%%
print("The confidence interval is:",model1.conf_int(alpha=0.05, cols=None))
print("The covariance matrix is:",model1.cov_params())
#%%
result = model1.predict(start=0,end=312)
true=np.array(y_test['first_high'])
error_11=true-result
y_test['ARMA11']=result
y_test.head()
#%%
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_11,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values of ARMA(1,1) model error')
plt.show()
#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q11=len(error_11)*np.sum(acf1**2)
print("The Q value is:",Q11)
#%%
from scipy.stats import chi2
DOF=20-2
alfa=0.01
chi_critical=chi2.ppf(1-alfa,DOF)
if Q11<chi_critical:
    print("the residual is white")
else:
    print("Not white")
#pass
#%%
#Pass zero/pole cancellation
np.roots([1,-1])
#%%
var_11=np.var(error_11)
mse_11=np.mean(error_11**2)
mean_11=np.mean(error_11)
rmse_11=(mse_11)**0.5
print("The Q value of ARMA(1,1) is:",Q11)
print("The variance of ARMA(1,1) is:",var_11)
print("The mse of ARMA(1,1) is:",mse_11)
print("The mean of ARMA(1,1) error is:",mean_11)
print("RMSE of ARMA(1,1) error is:",rmse_11)

#%%
#na=2,nb=1
model2=sm.tsa.ARMA(y,(2,1)).fit(trend='nc',disp=0)
print(model2.summary())
#%%
#Pass
np.roots([1,0])
#%%
print("The confidence interval is:",model2.conf_int(alpha=0.05, cols=None))
print("The covariance matrix is:",model2.cov_params())
#%%
result2 = model2.predict(start=0,end=312)
#result2[0]=y_test['first_high'][0]
true2=np.array(y_test['first_high'])
error_21=true2-result2
y_test['ARMA21']=result2
y_test.head()
#%%
acf=[]
for i in range(20):
    acf.append(get_auto_corr(error_21,i))
L1=np.arange(0,20,1)
L2=-L1[::-1]
x = np.concatenate((L2[0:-1], L1))
acf_reverse = acf[::-1]
ACF = np.concatenate ((acf_reverse[0:-1], acf))
plt.stem(x,ACF, use_line_collection=True, markerfmt = 'o')
plt.xlabel('lags')
plt.ylabel('ACF value')
plt.title('ACF for 20 values of ARMA(2,1) model error')
plt.show()
#%%
acf.remove(acf[0])
acf1=np.array(acf)
Q21=len(error_21)*np.sum(acf1**2)
DOF=20-3
alfa=0.01
chi_critical=chi2.ppf(1-alfa,DOF)
if Q21<chi_critical:
    print("the residual is white")
else:
    print("Not white")
#Pass
#%%
var_21=np.var(error_21)
mse_21=np.mean(error_21**2)
mean_21=np.mean(error_21)
rmse_21=(mse_21)**0.5
print("The Q value of ARMA(2,1) is:",Q21)
print("The variance of ARMA(2,1) is:",var_21)
print("The mse of ARMA(2,1) is:",mse_21)
print("The mean of ARMA(2,1) error is:",mean_21)
print("RMSE of ARMA(2,1) error is:",rmse_21)

#%%
#Pick ARMA(2,1) finally
#Left holt winter, regression, ARMA(2,1) models finally
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
data={'method':['Holt Winter','Regression','ARMA(2,1)'],
      'MSE':[mse_winter,mse_reg,mse_21],
      'mean':[mean_winter,mean_reg,mean_21],
      'variance':[var_winter,var_reg,var_21],
      'Q value':[Q_winter,Q_reg,Q21],
      'RMSE':[rmse_winter,rmse_reg,rmse_21]}
table=pd.DataFrame(data)
table
#%%
#plt.figure(figsize=(10,8))
plt.plot(y_test['first_high'], label='True',marker='o',markersize=8)
plt.plot(y_test['Holt_Winter'], label='Holt_Winter',marker='p',markersize=8)
plt.plot(y_test['Regression'],label='Regression',marker='+',markersize=8)
plt.plot(y_test['ARMA21'],label='ARMA(2,1)',marker='x',markersize=8)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Plot of predicted value versus true values')
plt.legend(loc='best')
plt.show()