
# coding: utf-8

# In[1]:


import theano

import theano.tensor as T
import numpy as np
import scipy as sp
import scipy.io as sio
import time
theano.config.floatX = 'float32'  

import lasagne.layers as ll

from lasagne.layers import InputLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import TransposedConv2DLayer

import lasagne.updates

from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import sigmoid

# max allowed disparity
d_max = 256;

# image dimensions
h = 370;
w = 1224;

# input variables
I_l_color_batch = T.tensor4(dtype='float32');
lr = T.scalar(dtype='float32');
sigma = T.scalar(dtype='float32');

E_total = T.tensor3(dtype='float32');     

D_gt = T.matrix('D_gt',dtype='int32'); 

nl_input = InputLayer(shape=(1, 3, h, w), input_var=I_l_color_batch);


# In[2]:


# this part is a modified version of the CNN based edge detector based on
# Holistically-nested edge detection, Xie et al. CVPR 2015

pad_sz = 34;
n_scales = 5;

# 1/2 of the feature maps is used compared to the original achitecture
# in order to improve the computational performance
mult = 0.5;

# scale 1
l_conv1_1 = ConvLayer(nl_input, name = "l_conv1_1", num_filters=64*mult, filter_size=3, pad=pad_sz+1,
                    nonlinearity=rectify, flip_filters=False);

l_conv1_2 = ConvLayer(l_conv1_1, name = "l_conv1_2", num_filters=64*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

conv1_2_out = ll.get_output(l_conv1_2).dimshuffle((0,1,3,2));

nl_conv_out = InputLayer(shape=(1,64*mult,None,None), input_var=conv1_2_out);

# transpose the image in order to circumvent the cuDNN restriction on the image size
l_pool1 = Pool2DLayer(nl_conv_out, name = 'l_pool1', stride=2, pool_size=2, mode='max', ignore_border=False);

pool1_out = ll.get_output(l_pool1).dimshuffle((0,1,3,2));

nl_pool_out = InputLayer(shape=(1,64*mult,None,None), input_var=pool1_out);

# scale 2
l_conv2_1 = ConvLayer(nl_pool_out, name = "l_conv2_1", num_filters=128*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv2_2 = ConvLayer(l_conv2_1, name = "l_conv2_2", num_filters=128*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_pool2 = Pool2DLayer(l_conv2_2, name = 'l_pool2', stride=2, pool_size=2, mode='max', ignore_border=False);
# scale 3
l_conv3_1 = ConvLayer(l_pool2, name = "l_conv3_1", num_filters=256*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv3_2 = ConvLayer(l_conv3_1, name = "l_conv3_2", num_filters=256*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv3_3 = ConvLayer(l_conv3_2, name = "l_conv3_3", num_filters=256*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_pool3 = Pool2DLayer(l_conv3_3, name = 'l_pool3', stride=2, pool_size=2, mode='max', ignore_border=False);
# scale 4
l_conv4_1 = ConvLayer(l_pool3, name = "l_conv4_1", num_filters=512*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv4_2 = ConvLayer(l_conv4_1, name = "l_conv4_2", num_filters=512*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv4_3 = ConvLayer(l_conv4_2, name = "l_conv4_3", num_filters=512*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_pool4 = Pool2DLayer(l_conv4_3, name = 'l_pool3', stride=2, pool_size=2, mode='max', ignore_border=False);
# scale 5
l_conv5_1 = ConvLayer(l_pool4, name = "l_conv5_1", num_filters=512*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv5_2 = ConvLayer(l_conv5_1, name = "l_conv5_2", num_filters=512*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

l_conv5_3 = ConvLayer(l_conv5_2, name = "l_conv5_3", num_filters=512*mult, filter_size=3, pad=1,
                    nonlinearity=rectify, flip_filters=False);

n_maps_upsample = 8;

# upsampling
l_score_dsn1 = ConvLayer(l_conv1_2, name = "l_score_dsn1", num_filters=n_maps_upsample, filter_size=1, pad=0, nonlinearity=None);

l_score_dsn2 = ConvLayer(l_conv2_2, name = "l_score_dsn2", num_filters=n_maps_upsample, filter_size=1, pad=0, nonlinearity=None);
l_upsample2 = TransposedConv2DLayer( l_score_dsn2, name = "l_upsample2",num_filters=n_maps_upsample, filter_size=4, stride=2, nonlinearity=None); 

l_score_dsn3 = ConvLayer(l_conv3_3, name = "l_score_dsn3", num_filters=n_maps_upsample, filter_size=1, pad=0, nonlinearity=None);
l_upsample4 = TransposedConv2DLayer(l_score_dsn3, name = "l_upsample4",num_filters=n_maps_upsample, filter_size=8, stride=4, nonlinearity=None);  

l_score_dsn4 = ConvLayer(l_conv4_3, name = "l_score_dsn4", num_filters=n_maps_upsample, filter_size=1, pad=0, nonlinearity=None);
l_upsample8 = TransposedConv2DLayer(l_score_dsn4, name = "l_upsample8",num_filters=n_maps_upsample, filter_size=16, stride=8, nonlinearity=None); 

l_score_dsn5 = ConvLayer(l_conv5_3, name = "l_score_dsn5", num_filters=n_maps_upsample, filter_size=1, pad=0, nonlinearity=None);
l_upsample16 = TransposedConv2DLayer(l_score_dsn5, name = "l_upsample16",num_filters=n_maps_upsample, filter_size=32, stride=16, nonlinearity=None);  

# multi-scale linear combination
# this code reproduces the original cropping behaviour of the caffe-hed code
# see https://github.com/s9xie/hed for details
edges1_matr = ll.get_output(l_score_dsn1)[:,:,pad_sz:pad_sz+h,pad_sz:pad_sz+w];
edges2_matr = ll.get_output(l_upsample2)[:,:,pad_sz+1:pad_sz+1+h,pad_sz+1:pad_sz+1+w];
edges3_matr = ll.get_output(l_upsample4)[:,:,pad_sz+2:pad_sz+2+h,pad_sz+2:pad_sz+2+w];
edges4_matr = ll.get_output(l_upsample8)[:,:,pad_sz+4:pad_sz+4+h,pad_sz+4:pad_sz+4+w];
edges5_matr = ll.get_output(l_upsample16)[:,:,pad_sz+8:pad_sz+8+h,pad_sz+8:pad_sz+8+w];

C = T.concatenate([edges1_matr,edges2_matr,edges3_matr,edges4_matr,edges5_matr],axis=1);

C_nl = InputLayer(shape=(1,n_scales*n_maps_upsample,h,w), input_var=C);

l_conv_final = ConvLayer(C_nl, name = "l_conv_final",num_filters=2, filter_size=1, pad=0, 
                         nonlinearity=sigmoid);


# In[3]:


### Trained Domain Transform (RNN guided by the edge detection CNN)

def rec_filter_func(x_t,w_t,y_tm1):
    y_t = (1 - w_t) * x_t + w_t * y_tm1;
    return y_t

def DT_horizontal(E_total, W_matrix_sig_h):
    E_filt_init_rot_clkws = (E_total[::-1]).dimshuffle((1,0,2));

    W_matrix_rot_clkws = (W_matrix_sig_h[::-1]).dimshuffle((1,0,2));

    E_filt_leftright, updates = theano.scan(fn=rec_filter_func, 
                                         sequences=[E_filt_init_rot_clkws[1:],W_matrix_rot_clkws[1:]],
                                         outputs_info = [dict(initial = E_filt_init_rot_clkws[0], taps = [-1])],
                                         n_steps=w-1, allow_gc = False)

    E_filt_leftright_fullsize = T.concatenate((E_filt_init_rot_clkws[np.newaxis,0,:,:],E_filt_leftright));

    E_filt_leftright_flipped = E_filt_leftright_fullsize[::-1];

    E_filt_rightleft, updates = theano.scan(fn=rec_filter_func, 
                                            sequences=[E_filt_leftright_flipped[1:],W_matrix_rot_clkws[-2::-1]],
                                            outputs_info = [dict(initial = E_filt_leftright_flipped[0], taps = [-1])],
                                            n_steps=w-1, allow_gc = False)

    E_filt_rightleft_fullsize = T.concatenate((E_filt_leftright_flipped[np.newaxis,0,:,:],E_filt_rightleft));

    E_filt_rightleft_fullsize_flipped = E_filt_rightleft_fullsize[::-1];

    E_filt_init = E_filt_rightleft_fullsize_flipped.dimshuffle((1,0,2))[::-1];
    
    return E_filt_init;

def DT_vertical(E_filt_init, W_matrix_sig_v):
    E_filt_updown, updates = theano.scan(fn=rec_filter_func, 
                                         sequences=[E_filt_init[1:],W_matrix_sig_v[1:]],
                                         outputs_info = [dict(initial = E_filt_init[0], taps = [-1])],
                                         n_steps=h-1, allow_gc = False)

    E_filt_updown_fullsize = T.concatenate((E_filt_init[np.newaxis,0,:,:],E_filt_updown));

    E_filt_updown_flipped = E_filt_updown_fullsize[::-1];

    E_filt_downup, updates = theano.scan(fn=rec_filter_func, 
                                         sequences=[E_filt_updown_flipped[1:],W_matrix_sig_v[-2::-1]],
                                         outputs_info = [dict(initial = E_filt_updown_flipped[0], taps = [-1])],
                                         n_steps=h-1, allow_gc = False)

    E_filt_downup_fullsize = T.concatenate((E_filt_updown_flipped[np.newaxis,0,:,:],E_filt_downup));

    E_filt_final = E_filt_downup_fullsize[::-1];
    
    return E_filt_final;


# In[4]:


# horizontal and vertical edge maps
edges_matr_v = (ll.get_output(l_conv_final))[0,0,0:h,0:w];
edges_matr_h = (ll.get_output(l_conv_final))[0,1,0:h,0:w];

edges_h = edges_matr_h[:,:,np.newaxis];
dt_w_h = T.exp(-1.0 * sigma * edges_h);

edges_v = edges_matr_v[:,:,np.newaxis];
dt_w_v = T.exp(-1.0 * sigma * edges_v);
   
W_matrix_sig_h = dt_w_h;
W_matrix_sig_v = dt_w_v;

E_input = E_total;

E_filt_hor = DT_horizontal(E_input,W_matrix_sig_h);
E_filt_vert = DT_vertical(E_filt_hor,W_matrix_sig_v);

E_output = E_filt_vert;

E_filt_final = E_output;

# Winner-takes-all disparity selection

D = T.zeros((h,w),dtype='int32');

for j in range(1,d_max):
    D = T.set_subtensor(D[:,j],(E_filt_final[:,j,0:j]).argmin(axis=1));
    
D = T.set_subtensor(D[:,d_max:],(E_filt_final[:,d_max:,:]).argmin(axis=2));

theano_disp_fn = theano.function(inputs=[I_l_color_batch,E_total,sigma], outputs=[D]);

# One-hot vector for each ground truth disparity

E_filt_lin = T.reshape(E_filt_final[:,:,:],(w*h,d_max+1));

E_filt_lin_sm = T.nnet.softmax(-1.0*E_filt_lin);

hw_index = range(0,w*h);

d_gt_center = D_gt[:,:].flatten();

D_gt_matr = T.zeros(((w)*h,d_max+1), dtype='float32');
D_gt_matr = T.inc_subtensor(D_gt_matr[(hw_index,d_gt_center)],1.0);

D_mask_vec = T.clip(d_gt_center,0,1);

# Cross-entropy loss
cross_entropy = T.nnet.categorical_crossentropy(E_filt_lin_sm,D_gt_matr);

L = (D_mask_vec*cross_entropy).mean();

list_layers = [l_conv_final, l_score_dsn1, l_upsample2, l_upsample4, l_upsample8, l_upsample16, l_pool1];
params_lasagne = []
for l in list_layers:
    params_lasagne = params_lasagne + ll.get_all_params(l);

params_lasagne = list(set(params_lasagne));

updates_lasagne = lasagne.updates.adam(L, params_lasagne, learning_rate=lr);

theano_train_fn = theano.function(inputs=[I_l_color_batch,E_total,D_gt,sigma,lr], 
                                  outputs=[L,D,edges_matr_v,edges_matr_h], 
                                  updates=updates_lasagne)


# In[5]:


mean_hed_rgb = np.array((104.00698793,116.66876762,122.67891434));

def load_cnn_weights(params_file):
    params = np.load(params_file)['names']
    param_names = [p for p in params]

    params = np.load(params_file)['values']
    param_vals = [p.astype(np.float32) for p in params]

    dict_vals = dict(zip(param_names, param_vals))

    for p in params_lasagne:
        v = dict_vals[p.name];
        
        if p.get_value().shape != v.shape:
            raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                             (p.get_value().shape, v.shape))
        else:
            p.set_value(v)


# In[6]:


# CUDA functions for the data term (census + SAD)
tensor_bnd_const_val = 18.35;
tensor_bnd_const_zero_val = 0;

sigma_val = 4.0;

coef_total = 120.0;
coef_sad = coef_total * 0.2 / 3.0 / 255.0;
coef_census = coef_total * 0.8 / 49.0 / 3.0;

n_el = h*w*(d_max+1);
n_ch_color = 3;
n_ch_gray  = 1;

# 7x7 census transform
wnd_half_census = 3;
# 1x1 sum of absolute differences
wnd_half_sad = 0;

import theano.misc.pycuda_init
import pycuda
import pycuda.driver as drv
import pycuda.gpuarray
from pycuda.compiler import SourceModule
import theano.sandbox.cuda as cuda_ndarray

mod_src_file = open('./cuda/stereo_utils.cu', 'r')
mod = SourceModule(mod_src_file.read())
    
census_func_pycuda = mod.get_function("census")
sad_func_pycuda = mod.get_function("sad_color")
linear_comb_func_pycuda = mod.get_function("linear_comb")
outlier_detection_func_pycuda = mod.get_function("outlier_detection");
interpolate_mismatch_func_pycuda = mod.get_function("interpolate_mismatch");
interpolate_occlusion_func_pycuda = mod.get_function("interpolate_occlusion");

def data_term(I_l_val,I_r_val,g_vol_sad,g_vol_census,g_vol_total):
    I_l_gray = rgb2gray(I_l_val);
    I_r_gray = rgb2gray(I_r_val);

    g_Il_color = cuda_ndarray.CudaNdarray(np.transpose(I_l_val,(2,0,1)).astype('float32'))
    g_Ir_color = cuda_ndarray.CudaNdarray(np.transpose(I_r_val,(2,0,1)).astype('float32'))
    g_Il_gray = cuda_ndarray.CudaNdarray(I_l_gray.astype('float32'))
    g_Ir_gray = cuda_ndarray.CudaNdarray(I_r_gray.astype('float32'))
    
    census_func_pycuda(
        g_Il_gray,g_Ir_gray,g_vol_census,np.int32(n_el),np.int32(n_ch_gray),np.int32(h),np.int32(w),
        np.int32(wnd_half_census),np.float32(tensor_bnd_const_zero_val), 
        block=(d_max+1,1,1), grid=(w, h))
    
    sad_func_pycuda(
        g_Il_color,g_Ir_color,g_vol_sad,np.int32(n_el),np.int32(h),np.int32(w),
        np.int32(wnd_half_sad),np.float32(tensor_bnd_const_val / coef_sad), 
        block=(d_max+1,1,1), grid=(w, h))

    linear_comb_func_pycuda(
        g_vol_sad,g_vol_census,g_vol_total,np.int32(n_el),np.float32(coef_sad),np.float32(coef_census), 
        block=(d_max+1,1,1), grid=(w, h))

    return;


# In[7]:


# auxiliary functions for KITTI 2015 data set

import scipy.ndimage
import os.path
import numpy as np
from scipy import misc
from skimage.color import rgb2gray
from scipy import ndimage

def load_disp(path):
    import scipy.ndimage
    return scipy.ndimage.imread(path)

def load_kitti_2015_img_color(data_dir, index, occluded=False, test=False, future=False):
    right = False
    path_left = os.path.join(data_dir,'data_scene_flow','testing' if test else 'training',
                             'image_3' if right else 'image_2',
                             "%06d_%2d.png" % (index, 11 if future else 10))
    
    right = True;
    path_right = os.path.join(data_dir,'data_scene_flow','testing' if test else 'training',
                             'image_3' if right else 'image_2',
                             "%06d_%2d.png" % (index, 11 if future else 10))
    
    path_disp = os.path.join(data_dir,'data_scene_flow','training',
                             'disp_occ_0' if occluded else 'disp_noc_0',
                             "%06d_%2d.png" % (index, 10))
    
    I_l = misc.imread(path_left);
    I_r = misc.imread(path_right);
    D_gt_val = (sp.ndimage.imread(path_disp)) / 256;
    
    I_l = I_l[0:h,0:w,:];
    I_r = I_r[0:h,0:w,:];
    D_gt_val = D_gt_val[0:h,0:w];
    
    I_l_color_val = I_l.astype(np.float32);
    I_r_color_val = I_r.astype(np.float32);
   
    I_l_orig = np.copy(I_l);
    I_r_orig = np.copy(I_r);
    
    I_l = I_l.astype(np.float32) - mean_hed_rgb;
    I_r = I_r.astype(np.float32) - mean_hed_rgb;
    
    I_l_color_batch_val = np.transpose(I_l[np.newaxis,:,:,:],(0,3,1,2));
    I_l_color_batch_val = np.float32(I_l_color_batch_val);
    
    I_r_color_batch_val = np.transpose(I_r[np.newaxis,:,:,:],(0,3,1,2));
    I_r_color_batch_val = np.float32(I_r_color_batch_val);
        
    return (I_l_color_batch_val,I_r_color_batch_val,I_l_orig,I_r_orig,D_gt_val);

def disp_error(D_gt_val,D_val):
    h = D_gt_val.shape[0];
    w = D_gt_val.shape[1];
    
    D_gt_mask_val = np.zeros((h,w));
    D_gt_mask_val[D_gt_val > 0] = 1.0;

    E = np.abs(D_val.astype('float32') - D_gt_val.astype('float32'));
    err_numer = (E*D_gt_mask_val > 3).sum();

    return float(err_numer) / max(((D_gt_val > 0.0).sum()),1)

def test_error(rng_test):
    n_img_test = len(rng_test);
    err_vec_full = np.zeros((n_img_test,1),dtype=np.float32);
    err_vec_leftright = np.zeros((n_img_test,1),dtype=np.float32);
    err_vec_med = np.zeros((n_img_test,1),dtype=np.float32);
    
    g_vol_census = cuda_ndarray.CudaNdarray.zeros((h,w,d_max+1));
    g_vol_sad = cuda_ndarray.CudaNdarray.zeros((h,w,d_max+1));
    g_vol_total = cuda_ndarray.CudaNdarray.zeros((h,w,d_max+1));
    
    for idx in range(0,n_img_test):
        i_img = rng_test[idx];
       
        occluded = True;
        [I_l_color_batch_val,I_r_color_batch_val,I_l_val,I_r_val,D_gt_val] = load_kitti_2015_img_color('/media/hpc3_storage/akuzmin/KITTI/',i_img, occluded);
 
        data_term(I_l_val,I_r_val,g_vol_sad,g_vol_census,g_vol_total);
        [D_left_val] = theano_disp_fn(I_l_color_batch_val,g_vol_total,sigma_val);

        data_term(I_r_val[:,::-1,:],I_l_val[:,::-1,:],g_vol_sad,g_vol_census,g_vol_total);
        [D_right_val] = theano_disp_fn(I_r_color_batch_val[:,:,:,::-1],g_vol_total,sigma_val);
        D_right_val = D_right_val[:,::-1];

        g_img_outlier = cuda_ndarray.CudaNdarray.zeros((h,w))
        g_d_left = cuda_ndarray.CudaNdarray(D_left_val.astype('float32'))
        g_d_right = cuda_ndarray.CudaNdarray(D_right_val.astype('float32'))

        outlier_detection_func_pycuda(
            g_d_left,g_d_right,g_img_outlier,np.int32(n_el),np.int32(w),np.int32(d_max), 
            block=(1,1,1), grid=(w, h))

        g_d_left_interp_occ = cuda_ndarray.CudaNdarray.zeros((h,w))
        interpolate_occlusion_func_pycuda(
            g_d_left,g_img_outlier,g_d_left_interp_occ,np.int32(n_el),np.int32(w), 
            block=(1,1,1), grid=(w, h))

        g_d_left_interp_mis = cuda_ndarray.CudaNdarray.zeros((h,w))
        interpolate_mismatch_func_pycuda(
            g_d_left_interp_occ,g_img_outlier,g_d_left_interp_mis,np.int32(n_el),np.int32(h),np.int32(w), 
            block=(1,1,1), grid=(w, h))
        
        D_left_interp = np.asarray(g_d_left_interp_mis);
        D_left_interp_med = ndimage.median_filter(D_left_interp, 5)
        
        erval_full = disp_error(D_gt_val,D_left_val);
        erval_leftright = disp_error(D_gt_val,D_left_interp);
        erval_med = disp_error(D_gt_val,D_left_interp_med);
       
        err_vec_full[idx] = erval_full;
        err_vec_leftright[idx] = erval_leftright;
        err_vec_med[idx] = erval_med;
    
    return (np.mean(err_vec_full),np.mean(err_vec_leftright),np.mean(err_vec_med))


# ## Apply the pretrained model

# In[8]:


load_cnn_weights('./pretrained_model/deep_cost_aggr_pretrained_kitti.npz');

rng_test = range(160,200);
[test_err_full,test_err_leftright,test_err_med] = test_error(rng_test);
            
print 'validation set error (occluded)'
print 'raw disparity error:', test_err_full
print 'error after the left-right check and median filter:', test_err_med
 


# ## Code for training

# In[9]:


from numpy import random

load_cnn_weights('./pretrained_model/hed_pretrained_bsds.npz');

rng_train = range(0,160);
rng_test = range(160,200);

n_iter = 10000;

mu = np.float32(1.0E-4); 

d_err_vec_test_raw = np.zeros((n_iter,1));
d_err_vec_test_leftright = np.zeros((n_iter,1));
d_err_vec_test_med = np.zeros((n_iter,1));

g_vol_census = cuda_ndarray.CudaNdarray.zeros((h,w,d_max+1));
g_vol_sad = cuda_ndarray.CudaNdarray.zeros((h,w,d_max+1));
g_vol_total = cuda_ndarray.CudaNdarray.zeros((h,w,d_max+1));

for i_iter in range(0,n_iter):
    if (np.mod(i_iter,10) == 0):
            print 'iter ', i_iter
            [test_err_raw,test_err_leftright,test_err_med] = test_error(rng_test);
            d_err_vec_test_raw[i_iter] = test_err_raw;
            d_err_vec_test_leftright[i_iter] = test_err_leftright;
            d_err_vec_test_med[i_iter] = test_err_med;
            
            print 'validation set error (occluded)'
            print 'raw disparity error:', test_err_raw
            print 'error after the left-right check and median filter:', test_err_med
    
    img_idx = random.randint(rng_train[0], rng_train[-1]) 

    [I_l_color_batch_val,I_r_color_batch_val,I_l_val,I_r_val,D_gt_val] = load_kitti_2015_img_color('/media/hpc3_storage/akuzmin/KITTI/',img_idx);
    #[I_l_color_batch_val,I_r_color_batch_val,I_l_val,I_r_val,D_gt_val] = load_kitti_2015_img_color('Z:/yy/stereo/',img_idx);

    data_term(I_l_val,I_r_val,g_vol_sad,g_vol_census,g_vol_total);
    
    # each single image is used as a batch for training as it contains many ground truth pixels
    [L_val,D_val,E_h_val,E_v_val] = theano_train_fn(I_l_color_batch_val,g_vol_total,D_gt_val,sigma_val,mu)
    

