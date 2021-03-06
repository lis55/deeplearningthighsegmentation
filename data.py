
import skimage.io as io
import random
import re
import shutil, os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import vtk
from vtk.util import numpy_support
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import SimpleITK as sitk

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def split(DATA_PATH, FRAME_PATH, MASK_PATH):
    # Create folders to hold images and masks
    folders = ['train_frames', 'train_masks', 'val_frames', 'val_masks', 'test_frames', 'test_masks']

    for folder in folders:
        if not os.path.isdir(DATA_PATH + '/' + folder):
            os.makedirs(DATA_PATH + '/' + folder)

    # Get all frames and masks, sort them, shuffle them to generate data sets.

    all_frames = os.listdir(FRAME_PATH)
    all_masks = os.listdir(MASK_PATH)

    all_frames.sort(key=lambda var: [int(x) if x.isdigit() else x
                                     for x in re.findall(r'[^0-9]|[0-9]+', var)])
    all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
                                    for x in re.findall(r'[^0-9]|[0-9]+', var)])

    random.seed(230)
    random.shuffle(all_frames)

    # Generate train, val, and test sets for frames

    train_split = int(0.7 * len(all_frames))
    val_split = int(0.9 * len(all_frames))

    train_frames = all_frames[:train_split]
    val_frames = all_frames[train_split:val_split]
    test_frames = all_frames[val_split:]

    # Generate corresponding mask lists for masks

    train_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in train_frames]
    val_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in val_frames]
    test_masks = [f for f in all_masks if 'image_' + f[6:16] + 'dcm' in test_frames]

    # Add train, val, test frames and masks to relevant folders


    def add_frames(dir_name, image):
        # img = Image.open(FRAME_PATH + image)
        # img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)
        shutil.move(FRAME_PATH + image, DATA_PATH + '/{}'.format(dir_name) + '/' + image)


    def add_masks(dir_name, image):
        # img = Image.open(MASK_PATH + image)
        # img.save(DATA_PATH + '/{}'.format(dir_name) + '/' + image)
        shutil.move(MASK_PATH + image, DATA_PATH + '/{}'.format(dir_name) + '/' + image)


    frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'),
                     (test_frames, 'test_frames')]

    mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'),
                    (test_masks, 'test_masks')]

    # Add frames

    for folder in frame_folders:
        array = folder[0]
        name = [folder[1]] * len(array)

        list(map(add_frames, name, array))

    # Add masks

    for folder in mask_folders:
        array = folder[0]
        name = [folder[1]] * len(array)

        list(map(add_masks, name, array))

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, data_path,
                 to_fit=True, batch_size=32, dim=(512, 512),
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.image_path = data_path + "/images"
        self.mask_path = data_path + "/masks"
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim,self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_dicom_image(self.image_path + '/'+ ID)
        #X=np.expand_dims(X, 4)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim,self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #y[i,] = self._load_grayscale_image_VTK(self.mask_path + '/'+'label_'+ID[6:15]+'.png')
            y[i,] = self._load_grayscale_image_VTK(self.mask_path + '/' + 'label_' + ID[6:15] + '.png')

        return y

    def _load_grayscale_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = img / 255
        '''
        img2 = img.astype(np.float32)

        # --- the following holds the square root of the sum of squares of the image dimensions ---
        # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
        value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))

        polar_image = cv2.linearPolar(img2, (img2.shape[0] / 2, img2.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)

        polar_image = polar_image.astype(np.uint8)
        img=polar_image
        '''
        img = np.expand_dims(img, axis=2)
        img=img.astype(np.float32)

        return img

    def _load_dicom_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = load_dicom(image_path)
        img = img / np.max(img)

        #self.polar(img)

        return img

    def _load_grayscale_image_VTK(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = vtk.vtkPNGReader()
        img.SetFileName(os.path.normpath(image_path))
        img.Update()

        _extent = img.GetDataExtent()
        ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

        img_data = img.GetOutput()
        datapointer = img_data.GetPointData()
        assert (datapointer.GetNumberOfArrays()==1)
        vtkarray = datapointer.GetArray(0)
        img = vtk.util.numpy_support.vtk_to_numpy(vtkarray)
        img = img.reshape(ConstPixelDims, order='F')

        img = img / np.max(img)
        img = img.astype('float32')

        #self.polar(img)

        return img

    def polar(self, img):
        img2 = img.astype(np.float32)

        # --- the following holds the square root of the sum of squares of the image dimensions ---
        # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
        value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))

        polar_image = cv2.linearPolar(img2, (img2.shape[0] / 2, img2.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)

        # polar_image = polar_image.astype(np.uint8)
        img = polar_image
        img = np.expand_dims(img, axis=2)

        return img

    def downsample(self,dim,path):
        ids = os.listdir(self.mask_path)
        for i in ids:
            img = self._load_grayscale_image_VTK(self.mask_path + '/' + i)[:, :, 0]
            img = img*255
            #img = img.astype(np.int8)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(path+'/'+'masks'+'/'+i, img)
        ids=os.listdir(self.image_path)
        for i in ids:
            img = self._load_dicom_image(self.image_path+'/'+i)[:,:,0]
            img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
            img = (img*255).astype(np.int16)
            #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(path+'/'+'images'+'/'+i[:-3]+'png', img)

def load_dicom(foldername, doflipz = True):
    reader = vtk.vtkDICOMImageReader()
    reader.SetFileName(foldername)
    reader.Update()

    #note: It workes when the OS sorts the files correctly by itself. If the files weren’t properly named, lexicographical sorting would have given a messed up array. In that
    # case you need to loop and pass each file to a separate reader through the SetFileName method, or you’d have to create a vtkStringArray, push the sorted filenames, and use
    # the vtkDICOMImageReader.SetFileNames method.

    # Load meta data: dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    # Get the 'vtkImageData' object from the reader
    imageData = reader.GetOutput()
    # Get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = imageData.GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays()==1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)

    # Convert the `vtkArray` to a NumPy array
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    # Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')


    return ArrayDicom

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2, test_frames_path=None, overlay=False, overlay_path=None):
    '''
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4]+".png"), img)
    '''
    if overlay:
        all_frames = os.listdir(test_frames_path)
        for i, item in enumerate(npyfile):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4] + ".png"), img)
            '''
            img2 = img.astype(np.float32)
            # --- the following holds the square root of the sum of squares of the image dimensions ---
            # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
            value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))
            polar_image = cv2.warpPolar(img2,img2.shape, (img2.shape[0] / 2, img2.shape[1] / 2), 800, cv2.WARP_FILL_OUTLIERS)
            polar_image = polar_image.astype(np.uint8)
            img = polar_image
            '''
            overlay = Image.fromarray((img*255).astype('uint8'))
            overlay = cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
            background = load_dicom(os.path.join(test_frames_path, all_frames[i]))
            background = background[:, :, 0] / np.max(background[:, :, 0])
            background = Image.fromarray((background * 255).astype('uint8'))
            background = background.convert("RGBA")
            overlay = overlay.convert("RGBA")

            # Split into 3 channels
            r, g, b, a = overlay.split()

            # Increase Reds
            g = b.point(lambda i: i * 0)

            # Recombine back to RGB image
            overlay = Image.merge('RGBA', (r, g, b, a))

            new_img = Image.blend(background, overlay, 0.3)
            # new_img = background
            new_img.save(os.path.join(overlay_path, 'image_' + all_frames[i][6:16] + 'png'), "PNG")
    else:
        for i, item in enumerate(npyfile):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
            io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4] + ".png"), img)



class DataGenerator2(DataGenerator):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, list_IDs, image_path, mask_path,
                 to_fit=True, batch_size=32, dim=(512, 512),
                 n_channels=1, n_classes=10, shuffle=True, data_gen_args=None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.bool = False
        if data_gen_args !=None:
            self.trans = ImageDataGenerator(**data_gen_args)
        super().__init__(list_IDs, image_path, mask_path,
                 to_fit, batch_size, dim,
                 n_channels, n_classes, shuffle)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        self.param = self.trans.get_random_transform(self.dim)

        if random.uniform(0,1) >= 0.5:
            self.bool = True
        else:
            self.bool = False
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_dicom_image(self.image_path + '/' + ID)
            # X[i,] = self.apply_transform(X[i,],self.get_random_transform((1,512,512)))
            if self.bool:
                X[i,] = self.trans.apply_transform(X[i,], self.param)


        # X=np.expand_dims(X, 4)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = self._load_grayscale_image_VTK(self.mask_path + '/' + 'label_' + ID[6:15] + '.png')
            if self.bool:
                y[i,] = self.trans.apply_transform(y[i,], self.param)

        return y


def plotFromGenerator3d(gen):
    count=0
    for i in gen:
      #pydicom.dcmread(gen(i))
      for batch in range(0,np.shape(i[0])[0]):
          for k in range(0,np.shape(i[0])[3]):
              background = i[0][batch,:,:,k,0]
              background = background/ np.max(background)
              background = (background * 255).astype('uint8')
              background = Image.fromarray(background)
              background = background.convert("RGBA")
              img = i[1][batch,:,:,k,0]
              overlay = Image.fromarray((img * 255).astype('uint8'))
              overlay = overlay.convert("RGBA")

              # Split into 3 channels
              r, g, b, a = overlay.split()

              # Increase Reds
              g = b.point(lambda i: i * 0)

              # Recombine back to RGB image
              overlay = Image.merge('RGBA', (r, g, b, a))
              new_img = Image.blend(background, overlay, 0.3)
              new_img.save(str(count)+ ".png", "PNG")
              count +=1
              '''  
              plt.imshow(new_img ,cmap=plt.cm.bone)
              plt.imsave("dicom.png", i[0][0, :, :, 0])
              plt.show()
              plt.imshow((i[1][0,:,:,0]),cmap=plt.cm.bone)
              plt.imsave("dicomlabel.png",i[1][0,:,:,0])
              plt.show()
              '''
class generator3da(Sequence):

    def __init__(self, list_IDs, image_path, mask_path,
                 to_fit=True, batch_size=32, patch_size=8, dim=(512, 512),dimy=(512,512),
                 n_channels=1, n_classes=10, shuffle=True, data_gen_args=None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.patch_size = patch_size
        self.number_of_patches=0
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.dimy = dimy
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
        slices = os.listdir(os.path.join(self.image_path, self.list_IDs[0]))
        self.number_of_patches = int(np.floor((len(slices)/self.patch_size)))
        self.patientIDs = list_IDs
        self.list_IDs = []
        if data_gen_args !=None:
            self.trans = ImageDataGenerator(**data_gen_args)

        temp = []
        count = 0


        for i,ID in enumerate(self.patientIDs):
            slices = os.listdir(os.path.join(self.image_path, ID))
            while count < self.number_of_patches:
                patch = slices[(count * self.patch_size):((count + 1) * self.patch_size)]
                self.list_IDs.append([ID,patch])
                count +=1
            count = 0
        self.indexes = np.arange(len(self.list_IDs))


    def _load_dicom_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = load_dicom(image_path)
        img = img / np.amax(img)

        #self.polar(img)

        return img

    def _load_grayscale_image_VTK(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = vtk.vtkPNGReader()
        img.SetFileName(os.path.normpath(image_path))
        img.Update()

        _extent = img.GetDataExtent()
        ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

        img_data = img.GetOutput()
        datapointer = img_data.GetPointData()
        assert (datapointer.GetNumberOfArrays()==1)
        vtkarray = datapointer.GetArray(0)
        img = vtk.util.numpy_support.vtk_to_numpy(vtkarray)
        img = img.reshape(ConstPixelDims, order='F')

        img = img / np.max(img)
        img = img.astype('float32')

        #self.polar(img)

        return img


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        #return int(np.floor(len(self.list_IDs) / self.batch_size) * self.number_of_patches)
        return int(np.floor(len(self.list_IDs)))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X


    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        Y = np.zeros((self.batch_size, *self.dimy, self.patch_size,self.n_channels))
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path= self.mask_path + '/' + patch[0] + '/' + 'label_' + ID[6:15] + '.png'
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                Y[0, :, :, i, 0] = img
                if self.bool:
                    Y[0,:,:,i,:] = self.trans.apply_transform(Y[0,:,:,i,:], self.param)
        return Y

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        X = np.zeros((self.batch_size, *self.dim, self.patch_size,self.n_channels))
        self.param = self.trans.get_random_transform(self.dim)

        if random.uniform(0,1) >= 0.5:
            self.bool = True
        else:
            self.bool = False
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path= self.image_path + '/' + patch[0] + '/' + ID
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                X[0, :, :, i, 0] = img
                if self.bool:
                    X[0,:,:,i,:] = self.trans.apply_transform(X[0,:,:,i,:], self.param)
        return X

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)





class generator3d(Sequence):

    def __init__(self, list_IDs, image_path, mask_path,
                 to_fit=True, batch_size=32, patch_size=8, dim=(512, 512),dimy =(512,512),
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.patch_size = patch_size
        self.number_of_patches=0
        self.list_IDs = list_IDs
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.dimy = dimy
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = 0
        self.max = self.__len__()
        slices = os.listdir(os.path.join(self.image_path, self.list_IDs[0]))
        self.number_of_patches = int(np.floor((len(slices)/self.patch_size)))
        self.patientIDs = list_IDs
        self.list_IDs = []

        temp = []
        count = 0


        for i,ID in enumerate(self.patientIDs):
            slices = os.listdir(os.path.join(self.image_path, ID))
            while count < self.number_of_patches:
                patch = slices[(count * self.patch_size):((count + 1) * self.patch_size)]
                self.list_IDs.append([ID,patch])
                count +=1
            count = 0
        self.indexes = np.arange(len(self.list_IDs))


    def _load_dicom_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = load_dicom(image_path)
        img = img / np.max(img)

        #self.polar(img)

        return img

    def _load_grayscale_image_VTK(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = vtk.vtkPNGReader()
        img.SetFileName(os.path.normpath(image_path))
        img.Update()

        _extent = img.GetDataExtent()
        ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

        img_data = img.GetOutput()
        datapointer = img_data.GetPointData()
        assert (datapointer.GetNumberOfArrays()==1)
        vtkarray = datapointer.GetArray(0)
        img = vtk.util.numpy_support.vtk_to_numpy(vtkarray)
        img = img.reshape(ConstPixelDims, order='F')

        img = img / np.max(img)
        img = img.astype('float32')

        #self.polar(img)

        return img


    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        #return int(np.floor(len(self.list_IDs) / self.batch_size) * self.number_of_patches)
        return int(np.floor(len(self.list_IDs)))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X


    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        Y = np.zeros((self.batch_size, *self.dimy, self.patch_size,self.n_channels))
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path= self.mask_path + '/' + patch[0] + '/' + 'label_' + ID[6:15] + '.png'
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                Y[0, :, :, i, 0] = img
        return Y

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        X = np.zeros((self.batch_size, *self.dim, self.patch_size,self.n_channels))
        # Generate data

        for patch in list_IDs_temp:
            for i, ID in enumerate(patch[1]):
                path= self.image_path + '/' + patch[0] + '/' + ID
                img = self._load_grayscale_image_VTK(path)[:, :, 0]
                X[0, :, :, i, 0] = img

        return X

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def load_grayscale_image_VTK(image_path):
    """Load grayscale image
    :param image_path: path to image to load
    :return: loaded image
    """
    img = vtk.vtkPNGReader()
    img.SetFileName(os.path.normpath(image_path))
    img.Update()


    _extent = img.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    img_data = img.GetOutput()
    datapointer = img_data.GetPointData()
    assert (datapointer.GetNumberOfArrays()==1)
    vtkarray = datapointer.GetArray(0)
    img = vtk.util.numpy_support.vtk_to_numpy(vtkarray)
    img = img.reshape(ConstPixelDims, order='F')

    img = img / np.max(img)
    img = img.astype('float32')

    #self.polar(img)

    return img


def load_mhd(filename):
    ##ImageDataGenerator expects (z,x,y) order of array
    itk_image = sitk.ReadImage(filename, sitk.sitkFloat32)
    np_array = sitk.GetArrayFromImage(itk_image)
    ##change order of dimensions to (x,y,z)
    # np_array = np.moveaxis(np_array, 0, -1)

    return (np_array)

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2, test_frames_path=None, overlay=False, overlay_path=None):
    '''
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4]+".png"), img)
    '''
    if overlay:
        all_frames = os.listdir(test_frames_path)
        for i, item in enumerate(npyfile):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4] + ".png"), img)
            '''
            img2 = img.astype(np.float32)
            # --- the following holds the square root of the sum of squares of the image dimensions ---
            # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
            value = np.sqrt(((img2.shape[0] / 2.0) ** 2.0) + ((img2.shape[1] / 2.0) ** 2.0))
            polar_image = cv2.warpPolar(img2,img2.shape, (img2.shape[0] / 2, img2.shape[1] / 2), 800, cv2.WARP_FILL_OUTLIERS)
            polar_image = polar_image.astype(np.uint8)
            img = polar_image
            '''
            overlay = Image.fromarray((img*255).astype('uint8'))
            background = load_dicom(os.path.join(test_frames_path, all_frames[i]))
            background = background[:, :, 0] / np.max(background[:, :, 0])
            background = Image.fromarray((background * 255).astype('uint8'))
            background = background.convert("RGBA")
            overlay = overlay.convert("RGBA")

            # Split into 3 channels
            r, g, b, a = overlay.split()

            # Increase Reds
            g = b.point(lambda i: i * 0)

            # Recombine back to RGB image
            overlay = Image.merge('RGBA', (r, g, b, a))

            new_img = Image.blend(background, overlay, 0.3)
            # new_img = background
            new_img.save(os.path.join(overlay_path, 'image_' + all_frames[i][6:16] + 'png'), "PNG")
    else:
        for i, item in enumerate(npyfile):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
            io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4] + ".png"), img)


def saveResult3dd(save_path,npyfile, patch_size =8, flag_multi_class = False,num_class = 2, test_frames_path=None):
    '''
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4]+".png"), img)
    '''
    all_frames = os.listdir(test_frames_path)
    count = 0
    test_data = []
    number_of_patches = np.floor(len(os.listdir(test_frames_path + '/' + all_frames[0])) / patch_size)
    for i, ID in enumerate(all_frames):
        slices = os.listdir(os.path.join(test_frames_path, ID))
        while count < number_of_patches:
            patch = slices[(count * patch_size):((count + 1) * patch_size)]
            test_data.append([ID, patch])
            count += 1
        count = 0

    for j,item in enumerate(npyfile):
        for i in range(0,np.shape(npyfile)[3]):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, i,0]
            print(j)
            io.imsave(os.path.join(save_path, test_data[j][1][i][:-4] + ".png"), img)


def saveResult3d( npyfile, patch_size=8, flag_multi_class=False, num_class=2, save_path = None, test_frames_path=None, framepath2=None, overlay_path=None,overlay_path2 = None):
    '''
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, os.listdir(test_frames_path)[i][:-4]+".png"), img)
    '''
    all_frames = os.listdir(test_frames_path)
    count = 0
    test_data = []
    number_of_patches = np.floor(len(os.listdir(test_frames_path + '/' + all_frames[0])) / patch_size)
    for i, ID in enumerate(all_frames):
        slices = os.listdir(os.path.join(test_frames_path, ID))
        while count < number_of_patches:
            patch = slices[(count * patch_size):((count + 1) * patch_size)]
            test_data.append([ID, patch])
            count += 1
        count = 0
    for j,item in enumerate(npyfile):
        for i in range(0,np.shape(npyfile)[3]):
            img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, i,0]
            io.imsave(os.path.join(save_path, test_data[j][1][i][:-4] + ".png"), img)
            imagepath = test_frames_path + '/' + test_data[j][0] + '/' + test_data[j][1][i]
            background = load_grayscale_image_VTK(imagepath)
            path = overlay_path+ '/'+ test_data[j][1][i][:-4] + '.png'
            overlay3d(background, img).save(path, "PNG")
            imagepath2 = framepath2 + '/'  + test_data[j][1][i][:-3]+'dcm'
            background = load_dicom(imagepath2)
            path = os.path.join(overlay_path2, test_data[j][1][i][:-4] + '.png')
            overlay3dup(background, img).save(path, "PNG")


def overlay3dup(background,overlay):
    img =background[:,:,0]
    img2 = img/ np.max(img)
    #background = Image.fromarray(img2)
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    background = Image.fromarray((img2 * 255).astype('uint8'))
    # background = background.rotate(90, expand=True)
    # background = Image.fromarray((img2).astype('float'))
    overlay = cv2.resize(overlay[:, :], (512, 512), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.medianBlur(overlay, 5)
    overlay = Image.fromarray((overlay * 255).astype('uint8'))


    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Split into 3 channels
    r, g, b, a = overlay.split()

    # Increase Reds
    g = b.point(lambda i: i * 0)

    # Recombine back to RGB image
    overlay = Image.merge('RGBA', (r, g, b, a))

    new_img = Image.blend(background, overlay, 0.3)
    return new_img
def overlay3d(background,overlay):
    img =background[:,:,0]
    img2 = img/ np.max(img)
    #background = Image.fromarray(img2)
    #img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    background = Image.fromarray((img2 * 255).astype('uint8'))
    # background = background.rotate(90, expand=True)
    # background = Image.fromarray((img2).astype('float'))
    overlay = Image.fromarray((overlay * 255).astype('uint8'))

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Split into 3 channels
    r, g, b, a = overlay.split()

    # Increase Reds
    g = b.point(lambda i: i * 0)

    # Recombine back to RGB image
    overlay = Image.merge('RGBA', (r, g, b, a))

    new_img = Image.blend(background, overlay, 0.3)
    return new_img

def overlay(save_path, image_path, mask_path):
    all_frames = os.listdir(image_path)
    all_masks = os.listdir(mask_path)
    for image,mask in zip(all_frames,all_masks):
        overlay = Image.open(os.path.join(mask_path,mask))
        img = load_dicom(os.path.join(image_path,image))
        img2=img[:,:,0]/np.max(img[:,:,0])
        background = Image.fromarray((img2*255).astype('uint8'))
        #background = background.rotate(90, expand=True)
        #background = Image.fromarray((img2).astype('float'))

        background = background.convert("RGBA")
        overlay = overlay.convert("RGBA")


        # Split into 3 channels
        r, g, b, a = overlay.split()

        # Increase Reds
        g = b.point(lambda i: i * 0)

        # Recombine back to RGB image
        overlay = Image.merge('RGBA', (r, g, b, a))

        new_img = Image.blend(background, overlay, 0.3)
        #new_img = background
        new_img.save(os.path.join(save_path,'image_' + image[6:16] + 'png'), "PNG")

def overlay3dupsample(save_path, image_path, mask_path):
    all_frames = os.listdir(image_path)
    all_masks = os.listdir(mask_path)
    for image, mask in zip(all_frames, all_masks):
        overlay = load_grayscale_image_VTK(os.path.join(mask_path, mask))
        overlay = cv2.resize(overlay[:,:,0],(512,512),interpolation=cv2.INTER_NEAREST)
        overlay = cv2.medianBlur(overlay, 5)
        overlay = Image.fromarray((overlay[:,:]*255).astype('uint8'))
        #overlay = Image.open(os.path.join(mask_path, mask))
        #overlay = overlay.resize((512,512))
        #overlay = overlay.filter(ImageFilter.MedianFilter(3))
        #overlay = cv2.medianBlur(overlay, 5)
        img = load_dicom(os.path.join(image_path, image))
        img2 = img[:, :,0] / np.max(img[:, :,0])
        background = Image.fromarray((img2 * 255).astype('uint8'))
        background = background.rotate(180, expand=False)
        # background = Image.fromarray((img2).astype('float'))

        background = background.convert("RGBA")
        overlay = overlay.convert("RGBA")

        # Split into 3 channels
        r, g, b, a = overlay.split()

        # Increase Reds
        g = b.point(lambda i: i * 0)

        # Recombine back to RGB image
        overlay = Image.merge('RGBA', (r, g, b, a))

        new_img = Image.blend(background, overlay, 0.3)
        # new_img = background
        new_img.save(os.path.join(save_path, 'image_' + image[6:16] + 'png'), "PNG")















