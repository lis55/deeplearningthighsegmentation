from data import *
from model import *
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

print(tensorflow.test.is_built_with_cuda())
print(tensorflow.test.gpu_device_name())

train_images_path = 'C:/fasciafilled/train_frames'
validation_images_path = 'C:/fasciafilled/val_frames'
test_images_path = "G:/DL_test_dataset_other_studies/elderly_women/images"
train_masks_path = 'C:/fasciafilled/train_masks'
validation_masks_path = 'C:/fasciafilled/val_masks'
test_masks_path = "G:/DL_test_dataset_other_studies/elderly_women/FASCIA_FINAL"


all_frames = os.listdir(train_masks_path)
gen = DataGenerator(all_frames, train_images_path, train_masks_path, to_fit=True,
                    batch_size=2, dim=(512, 512), n_channels=1, n_classes=1, shuffle=True)
all_frames = os.listdir(validation_images_path)
genAug = DataGenerator(all_frames, validation_images_path, validation_masks_path, to_fit=True, batch_size=2,
                       dim=(512, 512), n_channels=1, n_classes=1, shuffle=True)
all_frames = os.listdir(test_images_path)
testGene = DataGenerator(all_frames, test_images_path,
                         test_masks_path, to_fit=True, batch_size=1,
                         dim=(512, 512), n_channels=1, n_classes=1, shuffle=False)


model = unet(pretrained_weights="unet_ThighOuterSurface.hdf5")

#uncomment the next section to train the network
'''
model_checkpoint = ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
model_checkpoint2 = ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(gen, validation_data=genAug, validation_steps=205, steps_per_epoch=720, epochs=300,
                              callbacks=[model_checkpoint,
                                         model_checkpoint2])

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

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
plt.savefig('accuracy.png')

'''

results =  model.predict_generator(testGene, len(test_images_path), verbose=1)
saveResult("C:/results", results)

#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)

#Save overlay
overlay('C:/resultsoverlay', test_images_path, "C:/results")