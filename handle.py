from homofilter import *
import os

path_in = 'YaleB100×100'
path_out = 'Dataout'
# filename = '0101.bmp'

def handle(filename):
    im = Image.open(os.path.join(path_in, filename))
    im_out = HomoFilter(im, 2, 0.25, 1,300)
    im_out.save(os.path.join(path_out, filename))
    im_out.show()

def filter():
    im = gass_high_H(100,100,2,0.25,1,300)
    im = np.uint8(normalize(im))
    im = Image.fromarray(im,'L')
    im.save('filter.bmp')

def transform():
    filelist = os.listdir(path_in)
    for filename in filelist:
        if os.path.splitext(filename)[1] == '.bmp':
            handle(filename)


filter()
# transform()
handle('0247.bmp')