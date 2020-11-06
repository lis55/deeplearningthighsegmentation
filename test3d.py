
import shutil, os
from data import *
from model import *
import tensorflow
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""


test_paths = [['G:/Datasets/elderlymen1/3ddownsampled/test/images', 'G:/Datasets/elderlymen1/3dmedium/test/images' ,'G:/Datasets/elderlymen2/3ddownsampled/image', 'G:/Datasets/youngmen/3ddownsampled/image', 'G:/Datasets/elderlywomen/3ddownsampled/image'],
              ['G:/Datasets/elderlymen1/3ddownsampled/test/masks', 'G:/Datasets/elderlymen1/3dmedium/test/masks' ,'G:/Datasets/elderlymen2/3ddownsampled/mask', 'G:/Datasets/youngmen/3ddownsampled/mask', 'G:/Datasets/elderlywomen/3ddownsampled/mask']]

save_paths = [['C:/final_results/elderlymen1/3d', 'G:/Datasets/elderlymen1/3ddownsampled/test/images',
               'G:/Datasets/elderlymen1/2d/images','C:/final_results/elderlymen1/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
              ['C:/final_results/elderlymen1/3dmedium', 'G:/Datasets/elderlymen1/3dmedium/test/images',
               'G:/Datasets/elderlymen1/2d/images','C:/final_results/elderlymen1/3dmediumdown', 'C:/final_results/elderlymen1/3dmediumoverlay'],
              ['C:/final_results/elderlymen2/3d', 'G:/Datasets/elderlymen2/3ddownsampled/image',
               'G:/Datasets/elderlymen2/2d/images','C:/final_results/elderlymen2/3doverlaydown', 'C:/final_results/elderlymen2/3doverlay'],
              ['C:/final_results/youngmen/3d', 'G:/Datasets/youngmen/3ddownsampled/image',
               'G:/Datasets/youngmen/2d/images','C:/final_results/youngmen/3doverlaydown', 'C:/final_results/youngmen/3doverlay'],
              ['C:/final_results/elderlywomen/3d', 'G:/Datasets/elderlywomen/3ddownsampled/image',
               'G:/Datasets/elderlywomen/2d/images', 'C:/final_results/elderlywomen/3doverlaydown',
               'C:/final_results/elderlywomen/3doverlay']]
'''
test_frames = test_paths[0][1]
test_masks = test_paths[1][1]
all_frames = os.listdir(test_frames)

testGene = generator3d(all_frames, test_frames, test_masks, to_fit=True,batch_size=1, patch_size=8, dim=(128, 128),dimy =(128,128), n_channels=1, n_classes=1, shuffle=False)
print(len(testGene))
model = unet3d(pretrained_weights='G:/models/3d/1/unet_ThighOuterSurfaceval.hdf5', input_size=(128, 128,8, 1))
results =  model.predict_generator(testGene, len(testGene), verbose=1)

#saveResult3dd("C:/results3d", results, test_frames_path=test_images_path,overlay_path='C:/resultsoverlay3d')
saveResult3d(results,save_path=save_paths[1][0],test_frames_path=save_paths[1][1],framepath2=save_paths[1][2],overlay_path=save_paths[1][3],overlay_path2=save_paths[1][4])
#overlay3dupsample('C:/final_results/elderlywomen/3doverlay','G:/Datasets/elderlywomen/2d/images','C:/final_results/elderlywomen/3d')
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)



test_images_path = 'C:/fascia3dsmol/test/images'
test_masks_path = 'C:/fascia3dsmol/test/masks'
all_frames = os.listdir('C:/fascia3dsmol/test/images')
testGene = gen3da(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, dim=(128, 128), n_channels=1, n_classes=1, shuffle=False, data_gen_args=data_gen_args)
plotFromGenerator(testGene)

'''

data_gen_args = data_gen_args_dict = dict(shear_range=30,
                    rotation_range=50,
                    horizontal_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range = 0.1,
                    fill_mode='nearest')
train_images_path = 'G:/Datasets/elderlymen1/3dmedium/train/images'
train_masks_path = 'G:/Datasets/elderlymen1/3dmedium/train/masks'
all_frames = os.listdir('G:/Datasets/elderlymen1/3dmedium/train/images')
gen = generator3da(all_frames, train_images_path, train_masks_path, to_fit=True,batch_size=1, patch_size=4, dim=(256, 256), dimy =(256,256), n_channels=1, n_classes=1, shuffle=True, data_gen_args = data_gen_args)

val_images_path = 'G:/Datasets/elderlymen1/3dmedium/val/images'
val_masks_path = 'G:/Datasets/elderlymen1/3dmedium/val/masks'
all_frames = os.listdir('G:/Datasets/elderlymen1/3dmedium/val/images')
genAug = generator3d(all_frames, val_images_path, val_masks_path, to_fit=True,batch_size=1, patch_size=4, dim=(256, 256), dimy =(256,256), n_classes=1, shuffle=True)

test_images_path = 'G:/Datasets/elderlymen1/3dmedium/test/images'
test_masks_path = 'G:/Datasets/elderlymen1/3dmedium/test/masks'
all_frames = os.listdir('G:/Datasets/elderlymen1/3dmedium/test/images')
testGene = generator3d(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, patch_size=4, dim=(256, 256),dimy =(256,256), n_channels=1, n_classes=1, shuffle=False)

test_frames = test_paths[0][1]
test_masks = test_paths[1][1]
all_frames = os.listdir(test_frames)

testGene = generator3d(all_frames, test_frames, test_masks, to_fit=True,batch_size=1, patch_size=4, dim=(256, 256),dimy =(256,256), n_channels=1, n_classes=1, shuffle=False)
print(len(testGene))
model = unet3dd(pretrained_weights='unet_ThighOuterSurfaceval.hdf5', input_size=(256, 256,4, 1))
results =  model.predict_generator(testGene, len(testGene), verbose=1)

#saveResult3dd("C:/results3d", results, test_frames_path=test_images_path,overlay_path='C:/resultsoverlay3d')
saveResult3d(results,save_path=save_paths[1][0],test_frames_path=save_paths[1][1],framepath2=save_paths[1][3],overlay_path=save_paths[1][3],overlay_path2=save_paths[1][4])
#overlay3dupsample('C:/final_results/elderlywomen/3doverlay','G:/Datasets/elderlywomen/2d/images','C:/final_results/elderlywomen/3d')
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, len(testGene), verbose=0)
print(loss)
print(acc)



#model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5', input_size=(128, 128,8, 1))
model = unet3dd(input_size=(256, 256, 4,1))

model_checkpoint = ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint2 = ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(generator=gen, validation_data=genAug, validation_steps=49, steps_per_epoch=399, epochs=70, callbacks=[model_checkpoint2,model_checkpoint])

print(history.history.keys())
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.savefig('loss.png')

plt.figure()

plt.plot(history.history['dice_coefficient'])
plt.plot(history.history['val_dice_coefficient'])
plt.title('Model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
plt.savefig('accuracy.png')

test_frames = test_paths[0][1]
test_masks = test_paths[1][1]
all_frames = os.listdir(test_frames)

testGene = generator3d(all_frames, test_frames, test_masks, to_fit=True,batch_size=1, patch_size=4, dim=(256, 256),dimy =(256,256), n_channels=1, n_classes=1, shuffle=False)
print(len(testGene))
model = unet3dd(pretrained_weights='unet_ThighOuterSurfaceval.hdf5', input_size=(256, 256,4, 1))
results =  model.predict_generator(testGene, len(testGene), verbose=1)

#saveResult3dd("C:/results3d", results, test_frames_path=test_images_path,overlay_path='C:/resultsoverlay3d')
saveResult3d(results,save_path=save_paths[1][0],test_frames_path=save_paths[1][1],framepath2=save_paths[1][3],overlay_path=save_paths[1][3],overlay_path2=save_paths[1][4])
#overlay3dupsample('C:/final_results/elderlywomen/3doverlay','G:/Datasets/elderlywomen/2d/images','C:/final_results/elderlywomen/3d')
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, len(testGene), verbose=0)
print(loss)
print(acc)

