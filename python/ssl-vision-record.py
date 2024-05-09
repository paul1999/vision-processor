#!/usr/bin/env python3
import argparse
import bisect
import subprocess
import sys
import threading
import time
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import cv2
import yaml
from google.protobuf.json_format import MessageToDict
import proto.ssl_vision_wrapper_tracked_pb2 as ssl_vision_wrapper

from geom_publisher import open_multicast_socket
from record_ground_truth import recv_vision


def receive_vision(sock, detections):
    while True:
        received = recv_vision(sock)
        if 'detection' in received:
            vision_detections.append(received['detection'])


def receive_tracker(sock, detections):
    while True:
        message = sock.recv(65536)
        received = ssl_vision_wrapper.TrackerWrapperPacket()
        received.ParseFromString(message)
        detections.append(MessageToDict(received, preserving_proto_field_name=True)['tracked_frame'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SSL-Vision recorder')
    parser.add_argument('folder', help='Geometry configuration file')
    parser.add_argument('--vision', default="bin/vision", help='SSL-Vision binary')
    parser.add_argument('--tracker_wd', default="", help='Vision tracker working directory')
    parser.add_argument('--tracker', default="bin/autoReferee", help='Vision tracker binary')
    parser.add_argument('--vision_ip', default='224.5.23.2', help='Multicast IP address of the vision')
    parser.add_argument('--vision_port', type=int, default=10006, help='Multicast port of the vision tracker')
    parser.add_argument('--tracker_port', type=int, default=10010, help='Multicast port of the vision tracker')
    args = parser.parse_args()

    data_folder = Path(args.folder)
    vision_binary = Path(args.vision).absolute()
    ssl_vision_dir = data_folder / 'ssl-vision-config'
    ssl_vision_config = ssl_vision_dir / 'robocup-ssl.xml'

    vision_detections = []
    tracker_detections = []
    threading.Thread(target=receive_vision, name="Vision Receiver", args=[open_multicast_socket(args.vision_ip, args.vision_port), vision_detections], daemon=True).start()
    threading.Thread(target=receive_tracker, name="Tracker Receiver", args=[open_multicast_socket(args.vision_ip, args.tracker_port), tracker_detections], daemon=True).start()

    for video in list(data_folder.glob('*.mp4')) + list(data_folder.glob('*.png')):
        if video.name == 'field.png':
            continue

        print(f"Recording {video}")
        #TODO replace , with . in ssl vision config files
        config = ElementTree.parse(str(ssl_vision_config))
        #config.find(".//Var[@name='camera index']").text = str(cam_id)
        config.find(".//Var[@name='Video']/Var[@name='file']").text = str(video.absolute())
        for addr in config.findall(".//Var[@name='Multicast Address']"):
            addr.text = args.vision_ip
        config.find(".//Var[@name='Multicast Port']").text = str(args.vision_port)
        config.write(str(ssl_vision_config))

        #TODO <properties><useThreads>false</useThreads></properties>
        with subprocess.Popen([str(args.tracker), ], cwd=str(args.tracker_wd)) as tracker:
            time.sleep(1.0)
            with subprocess.Popen([str(vision_binary), '-s', '-c', '1'], cwd=str(ssl_vision_dir), stdout=subprocess.PIPE) as vision:
                while (line := vision.stdout.readline()) != b"End of video stream reached\n":
                    print(line)

                vision.terminate()
                vision.wait()
            time.sleep(1.0)
            tracker.terminate()
            tracker.wait()

        for t in tracker_detections:
            v_index = bisect.bisect_left(vision_detections, t['timestamp'], key=lambda v: v['t_capture'])
            v = vision_detections[v_index]
            if v_index+1 < len(vision_detections) and abs(t['timestamp'] - vision_detections[v_index+1]['t_capture']) < abs(t['timestamp'] - v['t_capture']):
                v = vision_detections[v_index+1]
            t['frame_number'] = v['frame_number']
            t['balls'] = [ball for ball in t['balls'] if ball['visibility'] > 0.0]

        if video.suffix == '.mp4':
            # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
            capture = cv2.VideoCapture(str(video))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            frames = 1

        if len(vision_detections) != frames or len(tracker_detections) != frames:
            print(f"Detection size mismatch: Expected {frames} Vision {len(vision_detections)} Tracker {len(tracker_detections)}")
            sys.exit(1)

        with video.with_suffix('.vision.yml').open('w') as file:
            yaml.dump(vision_detections, file)
        vision_detections.clear()

        with video.with_suffix('.tracker.yml').open('w') as file:
            yaml.dump(tracker_detections, file)
        tracker_detections.clear()
