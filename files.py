
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

train_images_path = 'C:/fascia3dsmol/train/images'
train_masks_path = 'C:/fascia3dsmol/train/masks'
all_frames = os.listdir('C:/fascia3dsmol/train/images')
gen = generator3da(all_frames, train_images_path, train_masks_path, to_fit=True,batch_size=1, patch_size=8, dim=(128, 128), n_channels=1, n_classes=1, shuffle=True, data_gen_args = data_gen_args)

val_images_path = 'C:/fascia3dsmol/val/images'
val_masks_path = 'C:/fascia3dsmol/val/masks'
all_frames = os.listdir('C:/fascia3dsmol/val/images')
genAug = generator3d(all_frames, val_images_path, val_masks_path, to_fit=True,batch_size=1, patch_size=8, dim=(128, 128), n_channels=1, n_classes=1, shuffle=True)

test_images_path = 'C:/fascia3dsmol/test/images'
test_masks_path = 'C:/fascia3dsmol/test/masks'
all_frames = os.listdir('C:/fascia3dsmol/test/images')
testGene = generator3d(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, patch_size=8, dim=(128, 128), n_channels=1, n_classes=1, shuffle=False)


model = unet3d(pretrained_weights='C:/model/unet_ThighOuterSurfaceval.hdf5', input_size=(128, 128,8, 1))
number_of_samples =len(os.listdir(test_images_path))*8*3
results =  model.predict_generator(testGene, 18, verbose=1)
saveResult3d("C:/results3d", results, test_frames_path=test_images_path,overlay_path='C:/resultsoverlay3d')
#saveResult3dd("C:/results3d", results, test_frames_path=test_images_path)
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)

#model = unet3d(pretrained_weights='unet_ThighOuterSurface.hdf5', input_size=(128, 128,8, 1))
model = unet3d(input_size=(128, 128, 8,1))

model_checkpoint = ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint2 = ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(generator=gen, validation_data=genAug, validation_steps=21, steps_per_epoch=216, epochs=1600, callbacks=[model_checkpoint2,model_checkpoint])

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

model = unet3d(pretrained_weights='unet_ThighOuterSurfaceval.hdf5', input_size=(128, 128,8, 1))
number_of_samples =len(os.listdir(test_images_path))*28
results =  model.predict_generator(testGene, number_of_samples, verbose=1)
saveResult3d("C:/results3d", results, test_frames_path=test_images_path,overlay_path='C:/resultsoverlay')
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)

