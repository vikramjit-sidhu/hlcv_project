"""
This module computes optic flow on a Video sequence 
It then stores this optic flow as an npy file
"""

import cv2
import numpy as np
import os
import scipy.misc
from PIL import Image
import random


bound = 20
parameter = 2
save_dir_type = 'data_dense'

def CheckVideo(video):
    e = -1 if len(video)%2 else -2
    tmp = video[0:e:2] - video[1:-1:2]
    tmp = tmp.sum(axis=3).sum(axis=2).sum(axis=1).astype(np.bool)
    ratio = 1 - tmp.mean()
    if ratio > 0.2:
        return video[0::2], ratio
    else:
        return video, ratio


def write_images_optic_flow(flows, data_root, video_id):
    for i, flow in enumerate(flows):
        result_path =  data_root + 'flow_img/frames/{:04}/'.format(video_id)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        imgpath =  result_path + '{:04}_flow1.png'.format(i)
        cv2.imwrite(imgpath, flow[:, :, 0])
        imgpath = result_path + '{:04}_flow2.png'.format(i)
        cv2.imwrite(imgpath, flow[:, :, 1])


def CalcFlow(video, data_root, video_id, parameter = 2):
    for i in range(0, len(video) - 2):
        #ipdb.set_trace()
        prev = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(video[i + 1], cv2.COLOR_BGR2GRAY)

        if parameter == 1:
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.702, 5, 10, 2, 7, 1.5,
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        elif parameter == 2:
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif parameter == 3:
            a=1
            #flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 3, 3, 5, 1.1, 0)#
        
        if i == 0:
            Flows = np.zeros((len(video) - 1,) + flow.shape, flow.dtype)
        Flows[i] = flow
    return Flows


def ConvertFlow2Img(Flows, lower_bound, higher_bound):
    DstFlows = (Flows - lower_bound) / float(higher_bound - lower_bound) * 255.
    DstFlows = np.round(DstFlows).astype(np.uint8)

    low_index = np.where(Flows < lower_bound)
    DstFlows[low_index] = 0
    high_index = np.where(Flows > higher_bound)
    DstFlows[high_index] = 255

    return DstFlows


def create_flow_npy_file_surreal():
    m_cap = 'cmu'
    data_root = '/data/unagi0/dataset/surreal/SURREAL/data/' + m_cap + '/'
    frame_root = '/data/unagi0/ohnishi/video_generation/human/data/surreal/' + m_cap + '/frames/'
    npy_root = '/data/unagi0/ohnishi/video_generation/human/data/surreal/' + m_cap + '/npy/'
    save_root_flow = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/surreal/' + m_cap + '/npy_flow/'
    save_root_small_img = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/surreal/' + m_cap + '/npy_76/'
    save_root_small_flow = '/data/unagi0/ohnishi/video_generation/human/' + save_dir_type + '/surreal/' + m_cap + '/npy_flow_76/'

    for data_type in ['train', 'test']:
        Data = []
        f = open(
            '/home/mil/ohnishi/workspace/video_generation/videogeneration_v2/data/surreal/' + data_type + '_modified.txt')
        for line in f.readlines():
            Data.append(line.split()[0])
        f.close()

        random.shuffle(Data)

        for data in Data:
            npy_path = npy_root + data_type + '/' + data + '.npy'
            save_path_flow = npy_path.replace(npy_root, save_root_flow)
            if not os.path.exists(save_path_flow):
                print save_path_flow
                video = np.load(npy_path)

                Flows = CalcFlow(video, parameter=parameter)
                Flows = ConvertFlow2Img(Flows, -1 * bound, bound)

                SmallImgs = np.zeros((video.shape[0], 76, 76, 3), np.uint8)
                SmallFlows = np.zeros((video.shape[0] - 1, 76, 76, 2), np.uint8)
                for i in range(video.shape[0]):
                    SmallImgs[i] = scipy.misc.imresize(video[i], [76, 76], 'bicubic')
                    if i < video.shape[0] - 1:
                        SmallFlows[i, :, :, 0] = scipy.misc.imresize(Flows[i, :, :, 0], [76, 76], 'bicubic')
                        SmallFlows[i, :, :, 1] = scipy.misc.imresize(Flows[i, :, :, 1], [76, 76], 'bicubic')

                save_path_small_flow = save_path_flow.replace(save_root_flow, save_root_small_flow)
                save_dir = os.path.split(save_path_small_flow)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path_small_flow, SmallFlows)

                save_path_small_img = npy_path.replace(npy_root, save_root_small_img)
                save_dir = os.path.split(save_path_small_img)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path_small_img, SmallImgs)

                save_dir = os.path.split(save_path_flow)[0]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                np.save(save_path_flow, Flows)


def create_flow_npy_file_penn_action():
    """
    This method creates npy files of the optic flow given a root data directory
    """

    data_root = '../../../Data/Penn_Action/frames'

    flow_root = '../../../Data/flows/'
    npy_root = '../../../Data/flows/npy/'
    save_root_flow = '../../../Data/flows/npy_flow/'
    save_root_small_img = '../../../Data/flows/npy_76/'
    save_root_small_flow = '../../../Data/flows/npy_flow_76/'

    # Number of videos in Penn Action
    num_videos = len(os.listdir(data_root))
    VideoIds = range(1, num_videos)
    # random.shuffle(VideoIds)

    for video_id in VideoIds:
        frame_path = data_root + '/{:04}/'.format(video_id)
        video = []
        num_frames = len(os.listdir(frame_path))

        # Load each of image sequence in the video into video
        for frame_id in range(1, num_frames + 1):
            img_path = frame_path + '{:06}.jpg'.format(frame_id)
            img = cv2.imread(img_path,1)
            video.append(img)

        Flows = CalcFlow(video, flow_root, video_id,  parameter=parameter)
        Flows = ConvertFlow2Img(Flows, -1 * bound, bound)
        write_images_optic_flow(Flows, flow_root, video_id)
        SmallImgs = np.zeros((len(video), 76, 76, 3), np.uint8)
        SmallFlows = np.zeros((len(video) - 1, 76, 76, 2), np.uint8)
        for i in range(len(video)):
            SmallImgs[i] = scipy.misc.imresize(video[i], [76, 76], 'bicubic')
            if i < len(video) - 1:
                SmallFlows[i, :, :, 0] = scipy.misc.imresize(Flows[i, :, :, 0], [76, 76], 'bicubic')
                SmallFlows[i, :, :, 1] = scipy.misc.imresize(Flows[i, :, :, 1], [76, 76], 'bicubic')

        if not os.path.exists(save_root_flow):
            os.makedirs(save_root_flow)
        save_path = save_root_flow + '{:04}.npy'.format(video_id)
        np.save(save_path, Flows)

        if not os.path.exists(save_root_small_img):
            os.makedirs(save_root_small_img)
        save_path = save_root_small_img + '{:04}.npy'.format(video_id)
        np.save(save_path, SmallImgs )
        # IN npy file flow dim is [num_frmaes-1, frame_size, 2], optical flow  data has 2 dim, one ofr flow along x adn one along y
        if not os.path.exists(save_root_small_flow):
            os.makedirs(save_root_small_flow)
        save_path = save_root_small_flow + '{:04}.npy'.format(video_id)
        np.save(save_path, SmallFlows)





# My code, not relevant anymore

def load_video_from_folder(folder_path, video_folder_name):
    full_filepath = os.path.join(folder_path, video_folder_name)
    video_frames_filenames = sorted(os.listdir(full_filepath))
    video = []
    # Each video frame is an image
    for video_frame_filename in video_frames_filenames:
        img = cv2.imread(video_frame_filename)
        video.append(img)
    return video


def get_all_videos_directory(folder_path):
    num_videos = len(os.listdir(folder_path))
    videos = []
    video_folder_names = sorted(os.listdir(folder_path))

    for folder_name in video_folder_names:
        video = load_video_from_folder(folder_path, folder_name)
        videos.append(video)
    return (num_videos, videos)


# def get_optic_flow_single_video(video):



def get_optic_flows_all_videos_as_images(videos):
    img_flows = []
    for video in videos:
        flow = get_optic_flow_single_video(video)
        img_flow = ConvertFlow2Img(flow)
        img_flows.append(img_flow)
    return img_flows





def create_npy_files_video_flow_penn_action():
    """
    Create npy files of optic flow from a dataset of images
    The structure is that there are various folders, each having a sequence of images
    """
    data_root = '../../../Data/Penn_Action/frames'

    # npy_root is where the npy file of the images are stored
    npy_root = '../../../Data/flows/npy/'
    # save_root_flow is where the npy file of the optic flow are stored
    save_root_flow = '../../../Data/flows/npy_flow/'

    # The same thing as the above for 
    save_root_small_img = '../../../Data/flows/npy_76/'
    save_root_small_flow = '../../../Data/flows/npy_flow_76/'

    num_videos, videos = get_all_videos_directory(data_root)
    optic_flows = get_optic_flows_all_videos_as_images(videos)



if __name__ == '__main__':
    create_flow_npy_file_penn_action()