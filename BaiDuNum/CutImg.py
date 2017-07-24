from numpy import *
from PIL import Image

for i in range(5000):
    path='E:\\image_contest_level_1\\gray\\'+str(i)+'.png'
    im=Image.open(path)
    print(i)
    for j in range(0,8):
        disc=(22.5*j,0,22.5*(j+1),60)
        path = 'E:\\image_contest_level_1\\gray\\cut\\' + str(i) + '_'+str(j)+'.png'
        imc=im.crop(disc)
        imc.save(path)

print('done')