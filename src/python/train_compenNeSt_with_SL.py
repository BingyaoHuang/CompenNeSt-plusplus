'''
Training and testing script for two-step-based CompenNeSt w/ SL (journal extension of cvpr'19 and iccv'19 papers)

This script trains/tests CompenNeSt w/ SL on different dataset specified in 'data_list' below.
The detailed training options are given in 'train_option' below.

1. We start by setting the training environment to GPU (if any).
2. K=20 setups are listed in 'data_list', which are our full compensation benchmark.
3. We set number of training images to 500 and loss function to l1+ssim, you can add other num_train and loss to 'num_train_list' and 'loss_list' for
comparison. Other training options are specified in 'train_option'.
4. The training data 'train_data' and validation data 'valid_data', are loaded in RAM using function 'loadData', and then we train the model with
function 'trainCompenNeStModel'. The training and validation results are both updated in Visdom window (`http://server:8098`) and console.
5. Once the training is finished, we can compensate the desired image. The compensation images 'prj_cmp_test' can then be projected to the surface.

Example:
    python train_compenNeSt_with_SL.py

See Models.py for CompenNeSt structure.
See trainNetwork.py for detailed training process.
See utils.py for helper functions.

Citation:
    @article{huang2020CompenNeSt++,
        title={End-to-end Full Projector Compensation},
        author={Bingyao Huang and Tao Sun and Haibin Ling},
        year={2021},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)} }

    @inproceedings{huang2019compennet++,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {CompenNet++: End-to-end Full Projector Compensation},
        booktitle = {IEEE International Conference on Computer Vision (ICCV)},
        month = {October},
        year = {2019} }

    @inproceedings{huang2019compennet,
        author = {Huang, Bingyao and Ling, Haibin},
        title = {End-To-End Projector Photometric Compensation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019} }
'''

# %% Set environment
import os

# set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device_ids = [0, 1, 2]

from trainNetwork import *
import Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() >= 1:
    print('Train with', len(device_ids), 'GPUs!')
else:
    print('Train with CPU!')

# %% K=20 setups
dataset_root = fullfile(os.getcwd(), '../../data')

data_list = [
    'light1/pos1/cloud_np',
    'light1/pos1/lavender_np',
    'light1/pos2/cubes_np',
    'light1/pos3/bars_spec_np',
    'light1/pos4/bubbles_np',
    'light1/pos5/pillow_np',
    'light2/pos1/curves_np',
    'light2/pos1/lavender_np',
    'light2/pos1/stripes_np',
    'light2/pos2/lavender_spec_np',
    'light2/pos3/curves_np',
    'light2/pos4/lavender_np',
    'light2/pos5/stripes_np',
    'light2/pos6/cubes_np',
    'light2/pos6/curves_np',
    'light3/pos1/bubbles_np',
    'light3/pos1/cloud_np',
    'light3/pos1/squares_np',
    'light3/pos2/curves_np',
    'light3/pos2/water_np',
]

# Training configurations of CompenNet++ reported in the paper
num_train_list = [500]
loss_list = ['l1+ssim']

# You can also compare different configurations, such as different number of training images and loss functions as shown below
# num_train_list = [8, 48, 125, 250, 500]
# loss_list = ['l1', 'l2', 'ssim', 'l1+l2', 'l1+ssim', 'l2+ssim', 'l1+l2+ssim']

# You can create your own models in Models.py and put their names in this list for comparisons.
model_list = ['CompenNeSt_with_SL']

# default training options
train_option_default = {'max_iters': 1500,
                        'batch_size': 48,
                        'lr': 1e-3,  # learning rate
                        'lr_drop_ratio': 0.2,
                        'lr_drop_rate': 1000,  # adjust this according to max_iters (lr_drop_rate < max_iters)
                        'loss': '',  # loss will be set to one of the loss functions in loss_list later
                        'l2_reg': 1e-4,  # l2 regularization
                        'device': device,
                        'pre-trained': False,
                        'plot_on': True,  # plot training progress using visdom (disable for faster training)
                        'train_plot_rate': 100,  # training and visdom plot rate (increase for faster training)
                        'valid_rate': 100}  # validation and visdom plot rate (increase for faster training)

# a flag that decides whether to compute and save the compensated images to the drive
save_compensation = True

# log file
from time import localtime, strftime

log_dir = '../../log'
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
log_file = open(fullfile(log_dir, log_file_name), 'w')
title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('data_name', 'model_name', 'loss_function',
                                'num_train', 'batch_size', 'max_iters',
                                'uncmp_psnr', 'uncmp_rmse', 'uncmp_ssim',
                                'valid_psnr', 'valid_rmse', 'valid_ssim'))
log_file.close()

# resize the input images if input_size is not None
input_size = None
# input_size = (256, 256) # we can also use a low-res input to reduce memory usage and speed up training/testing with a sacrifice of precision
resetRNGseed(0)

# stats for different setups
for data_name in data_list:
    # load training and validation data
    data_root = fullfile(dataset_root, data_name)
    cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask_corners = loadData(dataset_root, data_name, input_size, CompenNeSt_only=True)

    # surface image for training and validation
    cam_surf_train = cam_surf.expand_as(cam_train)
    cam_surf_valid = cam_surf.expand_as(cam_valid)

    # convert valid data to CUDA tensor if you have sufficient GPU memory (significant speedup)
    cam_valid = cam_valid.to(device)
    prj_valid = prj_valid.to(device)

    # validation data, 200 image pairs
    valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    # stats for different #Train
    for num_train in num_train_list:
        train_option = train_option_default.copy()
        train_option['num_train'] = num_train

        # select a subset to train
        train_data = dict(cam_surf=cam_surf_train[:num_train, :, :, :], cam_train=cam_train[:num_train, :, :, :],
                          prj_train=prj_train[:num_train, :, :, :])

        # stats for different models
        for model_name in model_list:

            train_option['model_name'] = model_name.replace('/', '_')

            # stats for different loss functions
            for loss in loss_list:
                log_file = open(fullfile(log_dir, log_file_name), 'a')

                # set seed of rng for repeatability
                resetRNGseed(0)

                # create a CompenNeSt
                compen_nest = Models.CompenNeSt()
                if torch.cuda.device_count() >= 1: compen_nest = nn.DataParallel(compen_nest, device_ids=device_ids).to(device)
                compen_nest = initCompenNeSt(compen_nest, dataset_root, device)

                # train option for current configuration, i.e., data name and loss function
                train_option['data_name'] = data_name.replace('/', '_')
                train_option['loss'] = loss

                print('-------------------------------------- Training Options -----------------------------------')
                print('\n'.join('{}: {}'.format(k, v) for k, v in train_option.items()))
                print('------------------------------------ Start training {:s} ---------------------------'.format(model_name))

                # train model
                compen_nest, valid_psnr, valid_rmse, valid_ssim = trainModel(compen_nest, train_data, valid_data, train_option)

                # uncompensated metrics
                cam_raw_valid = readImgsMT(fullfile(data_root, 'cam/raw/test'))
                cam_desire_valid = readImgsMT(fullfile(data_root, 'cam/desire/test'))
                uncmp_psnr, uncmp_rmse, uncmp_ssim = computeMetrics(cam_raw_valid, cam_desire_valid)

                # save results to log file
                ret_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
                log_file.write(ret_str.format(data_name, model_name, loss, num_train, train_option['batch_size'], train_option['max_iters'],
                                              uncmp_psnr, uncmp_rmse, uncmp_ssim,
                                              valid_psnr, valid_rmse, valid_ssim))
                log_file.close()

                # [testing phase] create compensated testing images
                if save_compensation:
                    print('------------------------------------ Start testing {:s} ---------------------------'.format(model_name))
                    torch.cuda.empty_cache()

                    # desired test images are created such that they can fill the optimal displayable area (see paper for detail)
                    desire_test_path = fullfile(data_root, 'cam/warpSL/desire/test')
                    assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)

                    # compensate and save images
                    desire_test = readImgsMT(desire_test_path).to(device)
                    cam_surf_test = cam_surf.expand_as(desire_test).to(device)
                    with torch.no_grad():
                        # simplify CompenNet
                        compen_nest.module.simplify(cam_surf_test[0, ...].unsqueeze(0))

                        # compensate using CompenNet
                        compen_nest.eval()
                        prj_cmp_test = compen_nest(desire_test, cam_surf_test).detach()  # compensated prj input image x^{*}
                    del desire_test, cam_surf_test

                    # create image save path
                    cmp_folder_name = '{}_{}_{}_{}_{}'.format(train_option['model_name'], loss, num_train, train_option['batch_size'],
                                                              train_option['max_iters'])
                    prj_cmp_path = fullfile(data_root, 'prj/cmp/test', cmp_folder_name)
                    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)

                    # save images
                    saveImgs(prj_cmp_test, prj_cmp_path)  # compensated testing images, i.e., to be projected to the surface
                    print('Compensation images saved to ' + prj_cmp_path)

                # clear cache
                del compen_nest
                torch.cuda.empty_cache()
                print('-------------------------------------- Done! ---------------------------\n')
        del train_data
    del cam_valid, prj_valid

print('All dataset done!')
