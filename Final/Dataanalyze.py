import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
对预处理好的数据进行分析
'''

class DataAnalyze:
    def __init__(self,filepath):
        self.filepath = filepath
        self.df = pd.read_excel(filepath,header=0,index_col=0)
        self.length = range(len(self.df))

    #for temp
    def paint_plot(self,titlename,filename,idx_list=[1,2]):
        for idx in idx_list:
            plt.plot(self.length,self.df.iloc[:,idx],label=self.df.columns[idx])
        plt.legend()
        plt.title(titlename)
        plt.savefig(filename)
        plt.show()

    #for rain and wind
    def paint_all_bar(self,titlename,filename,idx=0):
        counts = self.df.iloc[:,idx].value_counts()
        counts = counts.sort_index()
        counts = list(counts)
        print(self.df.iloc[:,idx].value_counts().sort_index(),counts)
        plt.bar(range(len(counts)),counts,color='limegreen')
        plt.xticks(range(len(counts)))
        plt.title(titlename)
        plt.savefig(filename)
        plt.show()

    #for rain and wind in month
    def paint_month_bar(self):
        keys=list(self.df.columns)[3:5]
        used_df = self.df[0:2191]
        #print(used_df)
        months = np.arange(1,13)
        width = 0.25
        ones_df = used_df[used_df[keys[0]]==1]
        sort_df=list(ones_df.iloc[:,0].value_counts().sort_index())
        plt.bar(months-width/2,sort_df,width=width,label='Rain',color='royalblue')
        ones_df = used_df[used_df[keys[1]]==1]
        sort_df=list(ones_df.iloc[:,0].value_counts().sort_index())
        plt.bar(months+width/2,sort_df,width=width,label='Wind',color='darkorange')
        plt.xticks(months)
        plt.legend()
        plt.title('Rain and wind days each month')
        plt.savefig('Data/rain_wind_month')
        plt.show()

#main
da = DataAnalyze('Data/data.xlsx')

da.paint_all_bar(
    titlename='Rainy, Cloudy, and Sunny',
    filename='Data/rain_cloud_sun',
    idx=3
)



