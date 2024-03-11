import os.path as osp
import pickle
import shutil
import tempfile
import os
import scipy.io as sio

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import cv2
import time

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

'''
def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results
'''

def single_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, iterNum = None):
    """Test with single GPU.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    print(tmpdir)
    dataset = data_loader.dataset
    if iterNum==None:
        output_mat_dir = os.path.join(tmpdir, 'mat')
        output_png_dir = os.path.join(tmpdir, 'png')
    else:
        output_mat_dir = os.path.join(tmpdir, str(iterNum), 'mat')
        output_png_dir = os.path.join(tmpdir, str(iterNum), 'png')

    print(output_mat_dir)
    if not os.path.exists(output_mat_dir):
        try:
            os.makedirs(output_mat_dir)
        except FileExistsError:
            pass
    print(output_png_dir)
    if not os.path.exists(output_png_dir):
        try:
            os.makedirs(output_png_dir)
        except FileExistsError:
            pass
    prog_bar = mmcv.ProgressBar(len(dataset))
    start_time = time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=False, **data)
            result = result.squeeze()
            print(torch.amax(result))
            print(torch.amin(result))
            print("---")
            sio.savemat(os.path.join(output_mat_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), {'result': result})
            png_res= 255*(1-result)
            cv2.imwrite(os.path.join(output_png_dir, '{}.png'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), png_res)
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    tm = time.time() - start_time
    print(tm)   


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, iterNum = None):
    """Test with single GPU.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.
    Returns:
        list: The prediction results.
    """
    world_size = 1
    model.eval()
    print(tmpdir)
    dataset = data_loader.dataset
    if iterNum==None:
        output_mat_dir = os.path.join(tmpdir, 'mat')
        output_png_dir = os.path.join(tmpdir, 'png')
    else:
        output_mat_dir = os.path.join(tmpdir, str(iterNum), 'mat')
        output_png_dir = os.path.join(tmpdir, str(iterNum), 'png')

    print(output_mat_dir)
    if not os.path.exists(output_mat_dir):
        try:
            os.makedirs(output_mat_dir)
        except FileExistsError:
            pass
    print(output_png_dir)
    if not os.path.exists(output_png_dir):
        try:
            os.makedirs(output_png_dir)
        except FileExistsError:
            pass
    prog_bar = mmcv.ProgressBar(len(dataset))
    start_time = time.time()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            #print(result.dtype)
            #print(result.size())
            result = result.squeeze()
            #print(np.sum(result), "SUM")
            #print(type(result))
            #print(result.size())
            #print(result.shape)
            #print(result.dtype)
            #print("---")
            sio.savemat(os.path.join(output_mat_dir, '{}.mat'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), {'result': result})
            #print(np.amax(result))
            #print(np.amin(result))
            png_res= np.round(255*result).astype(np.uint8)
            #print(data['img_metas'][-1].data[-1][-1]['img_id'])
            cv2.imwrite(os.path.join(output_png_dir, '{}.png'.format(data['img_metas'][-1].data[-1][-1]['img_id'])), png_res)
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    tm = time.time() - start_time
    print(tm) 


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
