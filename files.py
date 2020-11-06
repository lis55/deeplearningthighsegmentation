
import shutil, os
from data import *
from model import *
import tensorflow
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

data_gen_args = data_gen_args_dict = dict(shear_range=20,
                    rotation_range=20,
                    horizontal_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range = 0.1,
                    fill_mode='nearest')

def reorderframes(d2_frame_path,d2_mask_path,d3_frame_path,d3_mask_path,slices,patients):
    all_frames = os.listdir(d2_frame_path)
    all_masks = os.listdir(d2_mask_path)
    for i in range(1, patients+1):
        if not os.path.isdir(d3_frame_path + '/' + str(i)):
            os.makedirs(d3_frame_path + '/' + str(i))
    for i in range(1, patients+1):
        if not os.path.isdir(d3_mask_path + '/' + str(i)):
            os.makedirs(d3_mask_path + '/' + str(i))
    count = 1
    for i in range(0, len(all_frames), slices):
        print(i)
        for j in range(i, i + slices):
            shutil.move(d2_frame_path + '/' + all_frames[j], d3_frame_path + '/' + str(count) + '/' + all_frames[j])
            shutil.move(d2_mask_path + '/' + all_masks[j], d3_mask_path + '/' + str(count) + '/' + all_masks[j])
        count += 1
'''test_images_path = 'C:/fasciafilled/images'
test_masks_path = 'C:/fasciafilled/FASCIA_FILLED'
all_frames = os.listdir(test_images_path)


DATA_PATH = 'C:/fasciadownsample/image2'
all_masks = os.listdir('C:/fasciadownsample/image2')

for i in range(1,73):
    if not os.path.isdir('C:/fasciadownsample' + '/' + str(i)):
        os.makedirs('C:/fasciadownsample' + '/' + str(i))
count=1
for i in range(0,len(all_frames),28):
    print(i)
    for j in range(i,i+28):
        shutil.move(DATA_PATH +'/'+ all_masks[j], 'C:/fasciadownsample' + '/' + str(count) + '/' + all_masks[j])
    count += 1

testGene = DataGenerator(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, dim=(512, 512), n_channels=1, n_classes=1, shuffle=False)

testGene.downsample((512,512),'C:/fasciadownsample')

'''

test_paths = [['G:/Datasets/elderlymen1/2d/test_frames','G:/Datasets/elderlymen2/2d/images', 'G:/Datasets/youngmen/2d/images', 'G:/Datasets/elderlywomen/2d/images'],
              ['G:/Datasets/elderlymen1/2d/test_masks','G:/Datasets/elderlymen2/2d/FASCIA_FINAL', 'G:/Datasets/youngmen/2d/FASCIA_FINAL', 'G:/Datasets/elderlywomen/2d/FASCIA_FINAL']]

test_frames = test_paths[0][0]
test_masks = test_paths[1][0]

#test_frames = 'G:/Datasets/elderlymen1/2d/images'
#test_masks = 'G:/Datasets/elderlymen1/2d/FASCIA_FILLED'

#reorderframes('G:/Datasets/elderlymen1/3dmedium/images','G:/Datasets/elderlymen1/3dmedium/masks',
#              'G:/Datasets/elderlymen1/3dmedium/image','G:/Datasets/elderlymen1/3dmedium/mask',28,72)
test_frames = 'G:/Datasets/elderlymen1/3dmedium/test/images'
test_masks = 'G:/Datasets/elderlymen1/3dmedium/test/masks'
all_frames = os.listdir(test_frames)
testGene = DataGenerator(all_frames, test_frames, test_masks, to_fit=True, batch_size=1, dim=(512, 512), n_channels=1, n_classes=1, shuffle=False)
te = generator3da(all_frames, test_frames, test_masks, to_fit=True,batch_size=1, patch_size=2, dim=(256, 256),dimy =(256,256), n_channels=1, n_classes=1, shuffle=False,data_gen_args=data_gen_args)
plotFromGenerator3d(te)
testGene.downsample((256,256),'G:/Datasets/elderlymen1/3dmedium')
reorderframes('G:/Datasets/elderlymen1/3dmedium/images','G:/Datasets/elderlymen1/3dmedium/masks',
              'G:/Datasets/elderlymen1/3dmedium/image','G:/Datasets/elderlymen1/3dmedium/mask',28,72)


