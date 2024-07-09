import sys
import os

from copy import deepcopy
import numpy as np
# from scipy.spatial.transform import Rotation as R
import time
import logging
import open3d as o3d
import torch
import zmq
import yaml
import argparse
import subprocess
from LMP import *

def run_vlpart(image):
    affordance = 'open'
    #-----------

    API_key = ""
    mllm = GPT4V(api_key=API_key)
    object = mllm.get_response(image)
    print(object)
    lmp = LMP(image_path)
    query = 'affordance is ' + affordance + ' and ' + 'object is ' + object
    lmp(query)

def connect_client():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # we need to write 127.0.0.1 instead of localhost. It does not work otherwise.
    # socket.bind("tcp://127.0.0.1:5562")
    # in case between two pc
    socket.bind("tcp://*:5561")
    try:
        while True:
            flags=0
            copy=True
            track=False

            info = socket.recv_json(flags=flags)
            msgs = socket.recv_multipart()

            image_np = msgs[0]
            image_np_buffer = memoryview(image_np)
            image_np = np.frombuffer(image_np_buffer, dtype=info["image_np_dtype"]).reshape(info["image_nps_shape"])

            print("Keys of dictionary sent from client: ", info.keys())
            print("Shape of obj_pcd sent in dictionary: ", info["image_np_shape"])
            print("Shape of obj_pcd sent after reconstruction: ", image_np.shape)

            run_vlpart(image_np)
            masks = np.load('/home/qf/'+'mask.npy')
            print('masks shape', masks.shape)
            is_failure = b"False"
            if len(masks) == 0:
                is_failure = b"True"
            elif len(masks) > 1:
                masks = masks[0]
            # If client side in py2
            # message_dict = {'is_failure': is_failure,
            #                 'grasp_poses': [pose.flatten().tolist() for pose in grasp_poses]}
            # message_str = yaml.dump(message_dict)
            # socket.send(message_str.encode('utf-8'))

            # If client side in py3
            info = dict(
                masks_dtype=str(masks.dtype),
                masks_shape=masks.shape,
            )
            print(info)
            # print(grasp_poses)
            # In send_multipart function, there was error when strings are not sent
            # in a list. So, strings are sent like this: [is_failure] and [obj_class].
            socket.send_json(info, flags | zmq.SNDMORE)
            socket.send_multipart([is_failure], flags | zmq.SNDMORE, copy=copy, track=track)
            socket.send_multipart([masks], flags, copy=copy, track=track)

    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()

if __name__ == '__main__':
    connect_client()