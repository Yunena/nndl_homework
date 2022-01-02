import re
import numpy as np
import pandas as pd
'''
对获取的rawdata进行预处理
'''


class DataPreprocess:
    def __init__(self,ori_path,aim_path):
        self.ori_path = ori_path
        self.aim_path = aim_path

    #rawdata to dataframe, save to excel
    def raw2df(self):
        with open(self.ori_path,'r') as file:
            string = file.read()
        pat1 = re.compile('<div class="th200">.*</div>')
        nouse_pat1 = re.compile('<div class="th200">| </div>')
        pat2 = re.compile('<div class="th140">.*</div>')
        nouse_pat2 = re.compile('<div class="th140">|</div>')
        re1 = re.findall(pat1,string)
        for i in range(len(re1)):
            re1[i] = re.sub(nouse_pat1,'',re1[i])
        re2 = re.findall(pat2,string)
        for i in range(len(re2)):
            re2[i] = re.sub(nouse_pat2,'',re2[i])
        #print(re1)
        #print(re2)

        np1 = np.array(self.get_month(re1))
        np2 = np.array(re2).reshape(len(re1),4)
        np2 = np2.T
        np2[0]=self.get_temp(np2[0])
        np2[1]=self.get_temp(np2[1])
        np2[2]=self.get_rain(np2[2])
        np2[3]=self.get_wind(np2[3])
        np1.astype('int')
        np2.astype('int')
        final = np.vstack((np1,np2))
        df = pd.DataFrame(final.T,dtype=np.int)
        df.columns = pd.Series(['Month','MaxTemp','MinTemp','Rain','Wind'])
        self.df2excel(df)


    #process month
    def get_month(self,l):
        for i in range(len(l)):
            s = l[i].split('-')[1]
            l[i] = int(s)
        return l

    #process temp
    def get_temp(self,l):
        for i in range(len(l)):
            s = l[i].replace('℃','')
            l[i] = int(s)
        return l

    #process rain
    def get_rain(self,l):
        for i,s in enumerate(l):
            if not s.find('雨')==-1 or not s.find('雪')==-1:
                l[i] = 2
            elif not s.find('阴')==-1:
                l[i] = 1
            else:
                l[i] = 0
            #print(s,l[i])
        return l

    #process wind
    def get_wind(self,l):
        pattern = re.compile('[0-9]+')
        for i,s in enumerate(l):
            match = re.findall(pattern,s)
            if len(match)>0 and int(match[len(match)-1])>3:
                l[i] = 1
            else:
                l[i] = 0
        return l

    #dataframe to excel
    def df2excel(self,df):
        df.to_excel(self.aim_path)




# main
dpp = DataPreprocess('Data/raw_data.txt','Data/data.xlsx')
dpp.raw2df()
