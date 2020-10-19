#import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import os
#import pydicom
from matplotlib import pyplot, cm


#Funtions for reading different types of image data and converting them to numpy arrays in (z,x,y) order of dimensions

#Function for reading mhd files, filename is the dir to a mhd file, something like "../user/images/example.mhd" and in the same 
# directory should lie the belonging "example.raw" file
def load_mhd(filename):
    ##ImageDataGenerator expects (z,x,y) order of array
    itk_image = sitk.ReadImage(filename, sitk.sitkFloat32)
    np_array = sitk.GetArrayFromImage(itk_image)
    ##change order of dimensions to (x,y,z)
    #np_array = np.moveaxis(np_array, 0, -1)
    
    return (np_array)


#Function for reading DICOM files, foldername is the path to the folder which contains seperate DICOM slices
#PathDicom = "../../YHT001_2_sep/t1_tse_tra_p2_512_1_biasCorr/"
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


    return (ArrayDicom)

