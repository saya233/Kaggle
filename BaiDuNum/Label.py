import os
import pandas as pd

data=open('E:\\image_contest_level_1\\labels.txt')
#label_df=pd.DataFrame(columns=['label','one','two','three','four','five','six','seven'])
label_df=pd.read_csv('label.csv')
'''
a=1
while 1:
    f=data.readline()
    if f == '':
        break
    label=f.split(' ')[1].strip('\n').strip()
    formula=f.split(' ')[0].strip()
    one=formula[0]
    two=formula[1]
    three=formula[2]
    four=formula[3]
    five=formula[4]
    if len(formula)==7 :
        six=formula[5]
        seven=formula[6]
    elif len(formula)==5 :
        six=''
        seven=''
    else :
        len(formula)
        print('formula is not right')

    label_df.loc[a]={'label':label,'one':one,'two':two,'three':three,'four':four,'five':five,'six':six,'seven':seven}
    a=a+1
    print(a)
'''
dummies_one=pd.get_dummies(label_df['one'],prefix='one')
dummies_two=pd.get_dummies(label_df['two'],prefix='two')
dummies_three=pd.get_dummies(label_df['three'],prefix='three')
dummies_four=pd.get_dummies(label_df['four'],prefix='four')
dummies_five=pd.get_dummies(label_df['five'],prefix='five')
dummies_six=pd.get_dummies(label_df['six'],prefix='six')
dummies_seven=pd.get_dummies(label_df['seven'],prefix='seven')

#axis=1 对齐添加后的所有列，axis=0会将列的值添加到原有数据的后面
label_dummies=pd.concat([label_df,dummies_one,dummies_two,dummies_three,dummies_four,dummies_five,dummies_six,dummies_seven],axis=1)

#label_df.to_csv('label.csv',index=False,header=True)
label_dummies.to_csv('label_dummies.csv',index=False,header=True)
