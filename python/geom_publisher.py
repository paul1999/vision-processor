#!/usr/bin/env python3
import argparse
import socket
import struct
import threading
import time

import yaml

import proto.ssl_vision_wrapper_pb2 as ssl_vision_wrapper
import proto.ssl_vision_geometry_pb2 as ssl_vision_geometry


def open_multicast_socket(ip, port):
    # Adapted from https://stackoverflow.com/a/1794373 (CC BY-SA 4.0 by Gordon Wrigley)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.setsockopt(
        socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
        struct.pack("4sl", socket.inet_aton(ip), socket.INADDR_ANY)
    )
    return sock


def update_cameras(wrapper, sock: socket.socket):
    calib = wrapper.geometry.calib
    while True:
        received = ssl_vision_wrapper.SSL_WrapperPacket()
        received.ParseFromString(sock.recv(65536))
        if received.HasField('geometry'):
            for camera in received.geometry.calib:
                if calib[camera.camera_id].SerializeToString(deterministic=True) == camera.SerializeToString(deterministic=True):
                    continue

                calib[camera.camera_id].CopyFrom(camera)
                print(f"Updated camera {camera.camera_id} calibration.")


def dict_to_protobuf(buf, name, value):
    if type(value) is dict:
        for key, entry in value.items():
            dict_to_protobuf(getattr(buf, name) if type(name) is str else buf[name], key, entry)
    elif type(value) is list:
        l = getattr(buf, name)
        for entry in value:
            i = len(l)
            l.add()
            dict_to_protobuf(getattr(buf, name) if type(name) is str else buf[name], i, entry)
    else:
        if type(name) is str:
            setattr(buf, name, value)
        else:
            buf[name] = value


def config_bool(config, key):
    return key not in config or config[key]


def generate_default_lines(wrapper, config):
    lines = wrapper.geometry.field.field_lines

    default_lines = config['default_lines'] if 'default_lines' in config else {}
    field = config['field']
    thickness = field['line_thickness']
    half_length = field['field_length'] / 2
    half_width = field['field_width'] / 2

    def add_line(name, x1, y1, x2, y2, type=None):
        lines.add()
        line = lines[len(lines) - 1]
        line.name = name
        line.p1.x = x1
        line.p1.y = y1
        line.p2.x = x2
        line.p2.y = y2
        line.thickness = thickness
        line.type = ssl_vision_geometry.SSL_FieldShapeType.Value(type if type else line.name)

    add_line(   'TopTouchLine', -half_length,  half_width,  half_length,  half_width)
    add_line('BottomTouchLine', -half_length, -half_width,  half_length, -half_width)
    add_line(   'LeftGoalLine', -half_length, -half_width, -half_length,  half_width)
    add_line(  'RightGoalLine',  half_length, -half_width,  half_length,  half_width)

    if config_bool(default_lines, 'halfway'):
        add_line(    'HalfwayLine',            0, -half_width,            0,  half_width)

    if config_bool(default_lines, 'goal2goal'):
        add_line('CenterLine', -half_length, 0, half_length, 0)

    penalty_length = half_length - field['penalty_area_depth']
    half_penalty = field['penalty_area_width'] / 2

    add_line(           'LeftPenaltyStretch', -penalty_length, -half_penalty, -penalty_length,  half_penalty)
    add_line(          'RightPenaltyStretch',  penalty_length, -half_penalty,  penalty_length,  half_penalty)
    add_line(  'LeftFieldLeftPenaltyStretch',    -half_length,  half_penalty, -penalty_length,  half_penalty)
    add_line( 'LeftFieldRightPenaltyStretch',    -half_length, -half_penalty, -penalty_length, -half_penalty)
    add_line( 'RightFieldLeftPenaltyStretch',  penalty_length, -half_penalty,     half_length, -half_penalty)
    add_line('RightFieldRightPenaltyStretch',  penalty_length,  half_penalty,     half_length,  half_penalty)

    if config_bool(default_lines, 'centercircle'):
        arcs = wrapper.geometry.field.field_arcs
        arcs.add()
        arc = arcs[len(arcs)-1]
        arc.name = 'CenterCircle'
        arc.type = ssl_vision_geometry.SSL_FieldShapeType.Value(arc.name)
        arc.center.x = arc.center.y = 0.0
        arc.radius = field['center_circle_radius']
        arc.a1 = 0
        arc.a2 = 6.283185
        arc.thickness = thickness


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Geometry Publisher')
    parser.add_argument('--vision_ip', default='224.5.23.2', help='Multicast IP address of the vision')
    parser.add_argument('--vision_port', type=int, default=10006, help='Multicast port of the vision')
    parser.add_argument('--config', default='geometry.yml', help='Geometry configuration file')
    args = parser.parse_args()

    wrapper = ssl_vision_wrapper.SSL_WrapperPacket()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    dict_to_protobuf(wrapper.geometry, 'field', config['field'])
    dict_to_protobuf(wrapper.geometry, 'calib', config['calib'])
    dict_to_protobuf(wrapper.geometry, 'models', config['models'])
    generate_default_lines(wrapper, config)

    sock = open_multicast_socket(args.vision_ip, args.vision_port)

    threading.Thread(target=update_cameras, name="Update cameras", args=(wrapper, sock), daemon=True).start()

    while True:
        sock.sendto(wrapper.SerializeToString(), (args.vision_ip, args.vision_port))
        time.sleep(1.0)
