import caffe
import ipdb
import cv2
import numpy as np
import skimage

def crop_image(x, target_height=227, target_width=227):
    
    image = skimage.img_as_float(skimage.io.imread(x)).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

deploy = '/home/sagnik/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
model = '/home/sagnik/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
mean = '/home/sagnik/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'

class CNN(object):

    def __init__(self, deploy=deploy, model=model, mean=mean, batch_size=10, width=227, height=227):

        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.batch_size = batch_size
        self.net, self.tformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        caffe.set_mode_cpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)

        tformer = caffe.io.tformer({'data':net.blobs['data'].data.shape})
        tformer.set_transpose('data', (2,0,1))
        tformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        tformer.set_raw_scale('data', 255)
        tformer.set_channel_swap('data', (2,1,0))

        return net, tformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = 30
        batch_ctr = 0

        all_feats = np.zeros([len(image_list)] + layer_sizes,dtype='float16')

        for start, end in zip(range(0, iter_until, self.batch_size), \
                              range(self.batch_size, iter_until, self.batch_size)):

            image_batch_file = image_list[start:end]
            image_batch = np.array(map(lambda x: crop_image(x, target_width=self.width, target_height=self.height), image_batch_file))

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float16)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.tformer.preprocess('data', in_)

            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
            feats = out[layers]

            all_feats[start:end] = feats

            batch_ctr = batch_ctr + 1
            print "Batches processed till now : " + str(batch_ctr) + "\n"

        return all_feats
