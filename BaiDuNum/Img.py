from PIL import  Image

from sklearn.decomposition import PCA,IncrementalPCA
import numpy as np
from numpy import *

'''
for i in range(0,100000):
    #图片灰化
    
    path='E:\\image_contest_level_1\\'+str(i)+'.png'
    im=Image.open(path)
    im=im.convert('L')
    save_path='E:\\image_contest_level_1\\gray\\'+str(i)+'.png'
    im.save(save_path)
    print(i)
'''

#图片缩小
'''
for i in range(0,100000):
path='E:\\image_contest_level_1\\gray\\'+str(i)+'.png'
im=Image.open(path)
save_path='E:\\image_contest_level_1\\gray\\small\\'+str(i)+'.png'
im.thumbnail((120,40))
im.save(save_path)
print(i)
'''


#PCA
lst=[]
for i in range(100000):
    path = 'E:\\image_contest_level_1\\gray\\small\\zero\\' + str(i) + '.png'
    im=Image.open(path)
    a=array(im)
    a=a.flatten().tolist()
    lst.append(a)
    print(i)
print('done')
lst=array(lst)
#pca=PCA(n_components=0.5)
pca=IncrementalPCA(n_components=2000,batch_size=5000)
s=pca.fit_transform(lst)
s.tofile('E:\\data\\dbase')
print(s)



#图像底色归零
'''
for i in range(100000):
    path = 'E:\\image_contest_level_1\\gray\\small\\' + str(i) + '.png'
    im = Image.open(path)
    a=array(im)
    common=np.argmax(np.bincount(a.flatten()))
    a[a==common]=0
    ima=Image.fromarray(a)
    ima.save('E:\\image_contest_level_1\\gray\\small\\zero\\'+str(i)+'.png')
    print(i)
'''