import keras
import numpy
import xml.etree.ElementTree as ET
import theano.tensor as T
import os
from PIL import Image
from scipy import misc


def crop_detection(imPath,new_width=150,new_height=150,save=False,test=False):
    im = Image.open(imPath)
    im = im.resize((new_width,new_height),Image.ANTIALIAS)

    image_array = numpy.array(im)
    image_array = numpy.rollaxis(image_array,2,0)
    image_array = image_array/255.0
    image_array = image_array * 2.0 - 1.0

    if(test):
        image_array = (image_array + 1.0) / 2.0 * 225.0
        image_array = numpy.rollaxis(image_array,2,0)
        image_array = numpy.rollaxis(image_array,2,0)

        misc.imsave('recovered.jpg', image_array)

    if(save):
        return image_array,im
    else:
        return image_array



class objInfo():
    """
    objInfo saves the information of an object, including its class num, its cords
    """
    def __init__(self,x,y,h,w,class_num):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.class_num = class_num


class Cell():
    """
    A cell is a grid cell of an image, it has a boolean variable indicating whether there are any objects in this cell,
    and a list of objInfo objects indicating the information of objects if there are any
    """
    def __init__(self):
        self.has_obj = False
        self.objs = []

class image():
    """
    Args:
       side: An image is divided into side*side grids
    Each image class has two variables:
       imgPath: the path of an image on my computer
       bboxes: a side*side matrix, each element in the matrix is cell
    """
    def __init__(self,side,imgPath):
        self.imgPath = imgPath
        self.boxes = []
        for i in range(side):
            rows = []
            for j in range(side):
                rows.append(Cell())
            self.boxes.append(rows)

    def parseXML(self,xmlPath,labels,side):
        """
        Args:
          xmlPath: The path of the xml file of this image
          labels: label names of pascal voc dataset
          side: an image is divided into side*side grid
        """
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        for obj in root.iter('object'):
            class_num = labels.index(obj.find('name').text)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            h = ymax-ymin
            w = xmax-xmin
            #objif = objInfo(xmin/448.0,ymin/448.0,np.sqrt(ymax-ymin)/448.0,np.sqrt(xmax-xmin)/448.0,class_num)

            #which cell this obj falls into
            centerx = (xmax+xmin)/2.0
            centery = (ymax+ymin)/2.0
            newx = (150.0/width)*centerx
            newy = (150.0/height)*centery

            h_new = h * (150.0 / height)
            w_new = w * (150.0 / width)

            cell_size = 150.0/side
            col = int(newx / cell_size)
            row = int(newy / cell_size)
           # print "row,col:",row,col,centerx,centery

            cell_left = col * cell_size
            cell_top = row * cell_size
            cord_x = (newx - cell_left) / cell_size
            cord_y = (newy - cell_top)/ cell_size

            objif = objInfo(cord_x, cord_y, numpy.sqrt(h_new/150.0), numpy.sqrt(w_new/150.0),class_num)
            self.boxes[row][col].has_obj = True
            self.boxes[row][col].objs.append(objif)

def prepareBatch(start,end,vocPath,rootpath):
    """
    Args:
      start: the number of image to start
      end: the number of image to end
      imageNameFile: the path of the file that contains image names
      vocPath: the path of pascal voc dataset
    Funs:
      generate a batch of images from start~end
    Returns:
      A list of end-start+1 image objects
    """
    directories = filter(os.path.isdir, os.listdir(vocPath))
    imageList = []
    labels = os.listdir(vocPath)
    for directory in directories:
        for fn in next(os.walk(directory))[2]:
            path = os.path.join(directory, fn)
           # labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
            imgname = os.path.splitext(os.path.basename(path))
            xmlPath = rootpath+'/Annotations/'+imgname+'.xml'
            img = image(side=7, imgPath=path)
            img.parseXML(xmlPath, labels, 7)
            imageList.append(img)

    return imageList


def generate_batch_data(vocPath,batch_size,sample_number,rootpath):
    """
    Args:
      vocPath: the path of pascal voc data
      imageNameFile: the path of the file of image names
      batchsize: batch size, sample_number should be divided by batchsize
    Funcs:
      A data generator generates training batch indefinitely
    """
    class_num = 20
    #Read all the data once and dispatch them out as batches to save time2

    TotalimageList = prepareBatch(0,sample_number,vocPath,rootpath)

    while 1:
        batches = sample_number // batch_size
        for i in range(batches):
            images = []
            boxes = []
            sample_index = numpy.random.choice(sample_number,batch_size,replace=True)
            #sample_index = [3]
            for ind in sample_index:
                image = TotalimageList[ind]
                #print image.imgPath
                image_array = crop_detection(image.imgPath,new_width=150,new_height=150)
                #image_array = np.expand_dims(image_array,axis=0)

                y = []
                for i in range(7):
                    for j in range(7):
                        box = image.boxes[i][j]
                        '''
                        ############################################################
                        #x,y,h,w,one_hot class label vector[0....0],objectness{0,1}#
                        ############################################################
                        '''
                        if(box.has_obj):
                            obj = box.objs[0]

                            y.append(obj.x)
                            y.append(obj.y)
                            y.append(obj.h)
                            y.append(obj.w)

                            labels = [0]*20
                            labels[obj.class_num] = 1
                            y.extend(labels)
                            y.append(1) #objectness
                        else:
                            y.extend([0]*25)
                y = numpy.asarray(y)
                #y = np.reshape(y,[1,y.shape[0]])

                images.append(image_array)
                boxes.append(y)
            #return np.asarray(images),np.asarray(boxes)
            yield numpy.asarray(images), numpy.asarray(boxes)


def custom_loss(y_true,y_pred):
    '''
    Args:
      y_true: Ground Truth output
      y_pred: Predicted output
      The forms of these two vectors are:
      ######################################
      ## x,y,h,w,p1,p2,...,p20,objectness ##
      ######################################
    Returns:
      The loss caused by y_pred
    '''
    y1 = y_pred
    y2 = y_true
    loss = 0.0

    scale_vector = []
    scale_vector.extend([2]*4)
    scale_vector.extend([1]*20)
    scale_vector = numpy.reshape(numpy.asarray(scale_vector),(1,len(scale_vector)))

    for i in range(49):
        y1_piece = y1[:,i*25:i*25+24]
        y2_piece = y2[:,i*25:i*25+24]

        y1_piece = y1_piece * scale_vector
        y2_piece = y2_piece * scale_vector

        loss_piece = T.sum(T.square(y1_piece - y2_piece),axis=1)
        loss = loss + loss_piece * y2[:,i*25+24]
        loss = loss + T.square(y2[:,i*25+24] - y1[:,i*25+24])

    #loss = T.sum(loss)
    loss = T.sum(loss)
    return loss


class LossHistory(keras.callbacks.Callback):
    '''
    Use LossHistory to record loss
    '''
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def Acc(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    object_num = 0

    count = 0
    for image in imageList:
        count += 1
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=150,new_height=150)
        img = numpy.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]

        for i in range(49):
            preds = out[i*25:(i+1)*25]
            if(preds[24] > thresh):
                object_num += 1
                row = int(i/7)
                col = int(i%7)
                class_num = numpy.argmax(preds[4:24])

                #Ground Truth
                box = image.boxes[row][col]
                if(box.has_obj):
                    for obj in box.objs:
                        true_class = obj.class_num
                        if(true_class == class_num):
                            correct += 1
                        break


    return correct*1.0/object_num

def Recall(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    obj_num = 0
    count = 0
    for image in imageList:
        count += 1
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=150,new_height=150)
        img = numpy.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]
        #for each ground truth, see we have predicted a corresponding result
        for i in range(49):
            preds = out[i*25:i*25+25]
            row = int(i/7)
            col = int(i%7)
            box = image.boxes[row][col]
            if(box.has_obj):
                for obj in box.objs:
                    obj_num += 1
                    true_class = obj.class_num
                    #see what we predict
                    if(preds[24] > thresh):
                        predcit_class = numpy.argmax(preds[4:24])
                        if(predcit_class == true_class):
                            correct += 1
    return correct*1.0/obj_num

def MeasureAcc(model,sample_number,vocPath,imageNameFile):
    imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
    acc = Acc(imageList,model)
    re = Recall(imageList,model)

    return acc,re

class TestAcc(keras.callbacks.Callback):
    '''
    calculate test accuracy after each epoch
    '''

    def on_epoch_end(self,epoch, logs={}):
        #Save check points
        filepath = 'weights'+str(epoch)+'.hdf5'
        print('Epoch %03d: saving model to %s' % (epoch, filepath))
       # self.model.save_weights('data'filepath, overwrite=True)

        #Test train accuracy, only on 2000 samples
        vocPath = 'data/VOC2012'
        imageNameFile = vocPath+'/ImageSets/Segmentation/val.txt'
        sample_number = 200
        acc,re = MeasureAcc(self.model,sample_number,vocPath,imageNameFile)
        print ('Accuracy and recall on train data is: %3f,%3f'%(acc,re))

