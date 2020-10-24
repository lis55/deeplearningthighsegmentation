from data import *
from model import *
import tensorflow
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""


print(tensorflow.test.is_built_with_cuda())
print(tensorflow.test.gpu_device_name())

data_gen_args_dict = dict(shear_range=10,
                    rotation_range=20,
                    horizontal_flip=True,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    fill_mode='nearest')
test_paths = [['G:/Datasets/elderlymen1/2d/test_frames','G:/Datasets/elderlymen2/2d/images', 'G:/Datasets/youngmen/2d/images', 'G:/Datasets/elderlywomen/2d/images'],
              ['G:/Datasets/elderlymen1/2d/test_masks','G:/Datasets/elderlymen2/2d/FASCIA_FINAL', 'G:/Datasets/youngmen/2d/FASCIA_FINAL', 'G:/Datasets/elderlywomen/2d/FASCIA_FINAL']]
test_frames = test_paths[0][0]
test_masks = test_paths[1][0]
all_frames = os.listdir(test_frames)
testGene = DataGenerator(all_frames, test_frames,test_masks, to_fit=True, batch_size=1,dim=(512, 512), n_channels=1, n_classes=1, shuffle=False)

model = unet(pretrained_weights="G:/models/2d/unet_ThighOuterSurface.hdf5")
results =  model.predict_generator(testGene, len(os.listdir(test_frames)), verbose=1)
saveResult("C:/final_results/elderlymen1/2d", results, test_frames_path=test_frames,overlay=True,overlay_path='C:/final_results/elderlymen1/2doverlay')
#print accuracy and validation loss
loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)