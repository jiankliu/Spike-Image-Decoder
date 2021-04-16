import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from tensorflow import set_random_seed
import pdb
import matplotlib
from matplotlib import pyplot as plt
import sys
import skimage
import skimage.measure
from scipy.io import loadmat
from sklearn import preprocessing
import cv2


np.random.seed(1)
set_random_seed(2)

def dense_decoder(ncell):
    model = Sequential(name = 'dense')
    model.add(Dense(512, input_dim = ncell))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    #model.add(Dense(784))
    model.add(Activation('sigmoid'))
    
    #model.add(Permute((2, 1)))
    model.add(Reshape(target_shape = (64, 64, 1), name = 'dense_out'))

    return model

def AE(input_shape):
    model = Sequential(name = 'ae')

    #AE encoder
    model.add(Conv2D(64, (7, 7), strides = (2, 2), padding = 'same', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), strides = (2, 2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    #AE decoder
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
        
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), strides = (1, 1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (7, 7), strides = (1, 1), padding = 'same'))
    #ae_out = Activation('tanh')(ae)
    model.add(BatchNormalization(name = 'ae_out'))
    #ae_out = Cropping2D(cropping = ((3, 3), (3, 3)), name = 'ae_out')(ae)
    return model

def cal_performance(src_imgs, dst_imgs):
    src_imgs = src_imgs.astype('float32')
    dst_imgs = dst_imgs.astype('float32')

    img_num = src_imgs.shape[0]
    all_mse = np.zeros(img_num)
    all_psnr = np.zeros(img_num)
    all_ssim = np.zeros(img_num)

    for i in range(img_num):
        all_mse[i] = skimage.measure.compare_mse(src_imgs[i], dst_imgs[i])
        all_psnr[i] = skimage.measure.compare_psnr(src_imgs[i], dst_imgs[i])
        all_ssim[i] = skimage.measure.compare_ssim(src_imgs[i], dst_imgs[i], multichannel = True)

    return np.mean(all_mse), np.mean(all_psnr), np.mean(all_ssim)


if __name__ == '__main__':
    spike_dict = {'CIFAR10':(478.0, 0.0), 'CIFAR100':(478.0, 0.0), 'CIFAR10_RGB':(478.0, 0.0), 'CIFAR100_RGB':(478.0, 0.0)}

    # number of fmri signal
    ncell = 90
    input_x = Input(shape = (ncell,))

    # Load dataset
    handwriten_69=loadmat('data.mat')
    Y_train = handwriten_69['spkTrn']
    Y_test = handwriten_69['spkTest']
    X_train = handwriten_69['stimTrn']
    X_test = handwriten_69['stimTest']

    resolution = 64
    X_train = X_train.reshape([X_train.shape[0], 28, 28])
    X_test = X_test.reshape([X_test.shape[0], 28, 28])

    X_train_64 = np.zeros((X_train.shape[0], resolution, resolution))
    for iImg in range(X_train.shape[0]):
        X_train_64[iImg] = cv2.resize(X_train[iImg], (resolution, resolution))

    X_test_64 = np.zeros((X_test.shape[0], resolution, resolution))
    for iImg in range(X_test.shape[0]):
        X_test_64[iImg] = cv2.resize(X_test[iImg], (resolution, resolution))

    X_train = X_train_64.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test_64.reshape([X_test.shape[0], resolution, resolution, 1])

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.


    ## Normlization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)

    os.environ['CUDA_VISIBLE_DEVICES'] = '14'

    model_dense = dense_decoder(ncell)
    model_ae = AE((resolution, resolution, 1))

    dense_out = model_dense(input_x)
    ae_out = model_ae(dense_out)

    optimizer = keras.optimizers.Adam()

    #model_dense.trainable = False
    end2end_model = Model(input_x, ae_out)
    end2end_model.summary()
    end2end_model.compile(loss = 'mse', optimizer = optimizer)

    weight_dir = 'e2e_model'
    result_dir = 'e2e_result'

    multiout_model = Model(input_x, [dense_out, ae_out])

    for i in range(10):
        end2end_model.fit(Y_train, X_train, batch_size = 10, epochs = 30, validation_data = (Y_test, X_test) )
         
        pred_dense, pred_ae = multiout_model.predict(Y_test)
        mse, psnr, ssim = cal_performance(X_test, pred_ae)
        print('%dcell AE:\n\tmse:%f psnr:%f ssim:%f'%(ncell, mse, psnr, ssim))

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    end2end_model.save_weights(os.path.join(weight_dir, 'end2end_digit69_spk.h5'))

    #if not os.path.exists(result_dir):
    #    os.mkdir(result_dir)
    np.save(os.path.join(result_dir, 'end2end_digit69_spk.npy'), pred_ae)
    pred_dense_trn, pred_ae_trn = multiout_model.predict(Y_train)
    np.save(os.path.join(result_dir, 'end2end_digit69_spk_trn.npy'), pred_ae_trn)
    
    # visualization the reconstructed images
    X_reconstructed_mu = pred_ae
    n = 10
    for j in range(1):
        plt.figure(figsize=(12, 2))    
        for i in range(n):
            # display original images
            ax = plt.subplot(2, n, i +j*n*2 + 1)
            plt.imshow(np.rot90(np.fliplr(X_test[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstructed images
            ax = plt.subplot(2, n, i + n + j*n*2 + 1)
            #plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
            plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
        plt.savefig('e2eRec_spk.png', dpi=300)

    plt.close()

