
import shutil, os
from data import *
from model import *
import tensorflow
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

'''
DATA_PATH = 'C:/Users/lis/Desktop/downsize/image'
all_masks = os.listdir(DATA_PATH)


for i in range(1,73):
    if not os.path.isdir('C:/fascia3dsmol/images' + '/' + str(i)):
        os.makedirs('C:/fascia3dsmol/images' + '/' + str(i))
count=1
for i in range(0,len(all_masks),28):
    print(i)
    for j in range(i,i+28):
        shutil.move(DATA_PATH +'/'+ all_masks[j], 'C:/fascia3dsmol/images' + '/' + str(count) + '/' + all_masks[j])
    count += 1


train_images_path = 'C:/fasciafilled/images'
train_masks_path = 'C:/fasciafilled/FASCIA_FILLED'


all_frames = os.listdir(train_images_path)
gen = DataGenerator2(all_frames, train_images_path, train_masks_path, to_fit=True,batch_size=2, dim=(512, 512), n_channels=1, n_classes=1, shuffle=True)

gen.downsample((128,128),'C:/Users/lis/Desktop/downsize')
listOfIds = os.listdir('C:/fascia3d/images')
listOfIds2 = os.listdir('C:/fascia3d/masks')
path ='C:/fascia3d/images'
path2 ='C:/fascia3d/masks'
dim = (128,128,1)
batch_size = 1
gen = generator3d(listOfIds, listOfIds2, path, path2, dim,28)
plotFromGenerator(gen)


import pydicom
from pydicom.data import get_testdata_files

print(__doc__)

# FIXME: add a full-sized MR image in the testing data
filename = get_testdata_files('C:/Users/lis/Desktop/downsize/image_101001005.dcm')[0]
ds = pydicom.dcmread(filename)

# get the pixel information into a numpy array
data = ds.pixel_array
print('The image has {} x {} voxels'.format(data.shape[0],
                                            data.shape[1]))
data_downsampling = data[::8, ::8]
print('The downsampled image has {} x {} voxels'.format(
    data_downsampling.shape[0], data_downsampling.shape[1]))

# copy the data back to the original data set
ds.PixelData = data_downsampling.tobytes()
# update the information regarding the shape of the data array
ds.Rows, ds.Columns = data_downsampling.shape

# print the image information given in the dataset
print('The information of the data set after downsampling: \n')
print(ds)'''

'''
for i in gen:
    plt.imshow((i[0][0,:,:]), cmap=plt.cm.bone)
    plt.show()
    plt.imshow((i[1][0, :, :]), cmap=plt.cm.bone)
    plt.show()
'''

'''
test_images_path = 'C:/elderlywomen3d/images'
train_images_path = 'C:/elderlywomen3d/images'
train_masks_path = 'C:/elderlywomen3d/masks'
all_frames = os.listdir('C:/elderlywomen3d/images')
testGene = gen3d(all_frames, train_images_path, train_masks_path, to_fit=True,batch_size=1, dim=(128,128), n_channels=1, n_classes=1, shuffle=True)
'''
'''
test_images_path = 'C:/Users/lis/Desktop/downsize/image'
test_masks_path = 'C:/Users/lis/Desktop/downsize/mask'
all_frames = os.listdir('C:/Users/lis/Desktop/downsize/image')
'''

data_gen_args = data_gen_args_dict = dict(shear_range=30,
                    rotation_range=20,
                    horizontal_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range = 0.1,
                    fill_mode='nearest')
test_images_path = 'C:/fascia3dsmol/test/images'
test_masks_path = 'C:/fascia3dsmol/test/masks'
all_frames = os.listdir('C:/fascia3dsmol/test/images')
testGene = gen3da(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, dim=(128, 128), n_channels=1, n_classes=1, shuffle=False, data_gen_args=data_gen_args)
plotFromGenerator(testGene)

test_images_path = 'C:/fascia3dsmol/test/images'
test_masks_path = 'C:/fascia3dsmol/test/masks'
all_frames = os.listdir('C:/fascia3dsmol/test/images')

test_images_path = 'C:/Users/lis/Desktop/downsize/image'
test_masks_path = 'C:/Users/lis/Desktop/downsize/mask'
all_frames = os.listdir('C:/Users/lis/Desktop/downsize/image')
testGene = gen3d(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, dim=(128, 128), n_channels=1, n_classes=1, shuffle=False)
model = unet3d(pretrained_weights='unet_ThighOuterSurfaceval.hdf5', input_size=(128, 128,8, 1))
results =  model.predict_generator(testGene, len(os.listdir(test_images_path)), verbose=1)
saveResult3d("C:/results3d", results, test_frames_path=test_images_path,overlay=True,overlay_path='C:/resultsoverlay3d')
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)