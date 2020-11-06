from data import *
from model import *
import tensorflow
import matplotlib.pyplot as plt
from PIL import ImageFilter


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

def accuracy(mask1_path, mask2_path):
    all_masks1 = os.listdir(mask1_path)
    all_masks2 = os.listdir(mask2_path)
    for mask1, mask2 in zip(all_masks1, all_masks2):
        m1 = load_grayscale_image_VTK(os.path.join(mask1_path, mask1))
        m2 = load_grayscale_image_VTK(os.path.join(mask2_path, mask2))
        m1 = cv2.resize(m1,(512,512),interpolation=cv2.INTER_NEAREST)
        m1 = cv2.medianBlur(m1, 5)
        print(tf.keras.backend.print_tensor(dice_coefficient(m1,m2),message=''))

def calc_dice(image_1, image_2):
    size = image_1.GetSize()
    spac = image_1.GetSpacing()
    unit_vol = spac[0]*spac[1]*spac[2]
    vol_1 = 0
    vol_2 = 0
    intersec = 0
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                hit_1 = image_1.GetPixel((x,y,z)) > 0
                if hit_1:
                    vol_1 += unit_vol
                hit_2 = image_2.GetPixel((x,y,z)) > 0
                if hit_2:
                    vol_2 += unit_vol
                if hit_1 and hit_2:
                    intersec += unit_vol
    dice = 2*intersec/(vol_1+vol_2)
    return (vol_1, vol_2, dice)


def dice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:
        intersection = np.logical_and(img, img2)
        value = (2. * intersection.sum()) / (img.sum() + img2.sum())
    return value

def calc_hausdorff(imagepath_1, imagepath_2):
    all_masks1 = os.listdir(imagepath_1)
    all_masks2 = os.listdir(imagepath_2)
    ''' 
    all_masks2 = []

    for i in all_masks1:
        all_masks2.append('label'+i[5:len(i)])
    '''
    hd=[]
    dice =[]
    for mask1, mask2 in zip(all_masks1,all_masks2):
        m1 = load_grayscale_image_VTK(os.path.join(imagepath_1, mask1))
        m1 = cv2.rotate(m1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        m1 = cv2.rotate(m1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #m1 = cv2.rotate(m1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        plt.imshow(Image.fromarray(m1[:,:]*255), cmap=plt.cm.bone)
        plt.show()
        m2 = load_grayscale_image_VTK(os.path.join(imagepath_2, mask2))
        #m2 = cv2.rotate(m2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plt.imshow(Image.fromarray(m2[:,:,0]*255), cmap=plt.cm.bone)
        plt.show()
        m1 = cv2.resize(m1[:, :], (512, 512), interpolation=cv2.INTER_NEAREST)
        m1 = cv2.medianBlur(m1, 5)
        dice.append(dice_coef(m1, m2[:,:,0]))
        m1 = sitk.GetImageFromArray(m1, isVector=False)
        m2 = sitk.GetImageFromArray(m2[:,:,0], isVector=False)
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(m1, m2)
        hd.append(hausdorff.GetHausdorffDistance())


    hd=np.array(hd)
    dice = np.array(dice)
    stat = [np.mean(hd),np.std(hd), np.max(hd),np.min(hd)]
    stat2 = [np.mean(dice), np.std(dice), np.max(dice), np.min(dice)]
    print (stat)
    print(stat2)
    return



calc_hausdorff('C:/final_results/elderlymen2/3d','G:/Datasets/elderlymen2/2d/FASCIA_FINAL')

accuracy('C:/results3d','C:/fascia3dtest/mask2')

test_images_path = 'C:/fascia3dsmol/test/images'
test_masks_path = 'C:/fascia3dsmol/test/masks'
all_frames = os.listdir('C:/fascia3dsmol/test/images')
testGene = generator3d(all_frames, test_images_path, test_masks_path, to_fit=True,batch_size=1, patch_size=8, dim=(128, 128),dimy =(128,128), n_channels=1, n_classes=1, shuffle=False)

model = unet3d(pretrained_weights='unet_ThighOuterSurfaceval.hdf5', input_size=(128, 128,8, 1))
results =  model.predict_generator(testGene, 6, verbose=1)
#saveResult3dd("C:/results3d", results, test_frames_path=test_images_path,overlay_path='C:/resultsoverlay3d')
saveResult3dd("C:/results3d", results, test_frames_path=test_images_path)
overlay3dupsample('C:/resultsoverlay3d','C:/fascia3dtest/overlay/images','C:/results3d')

loss, acc = model.evaluate_generator(testGene, steps=3, verbose=0)
print(loss)
print(acc)