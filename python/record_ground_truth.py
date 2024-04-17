#!/usr/bin/env python3
import argparse
import socket
import struct
import sys

import yaml
from geom_publisher import open_multicast_socket
from google.protobuf.json_format import MessageToDict

import proto.ssl_vision_wrapper_pb2 as ssl_vision_wrapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Geometry Publisher')
    parser.add_argument('--source_ip', help='Optional source IP address filter')
    parser.add_argument('--vision_ip', default='224.5.23.2', help='Multicast IP address of the vision')
    parser.add_argument('--vision_port', type=int, default=10006, help='Multicast port of the vision')
    parser.add_argument('--gt', default='gt.yml', help='Output ground truth')
    parser.add_argument('--geom', default='gt-geometry.yml', help='Output geometry')
    args = parser.parse_args()

    detection = False
    geometry = False
    sock = open_multicast_socket(args.vision_ip, args.vision_port)
    while True:
        message, address = sock.recvfrom(65536)
        if args.source_ip and args.source_ip != address[0]:
            continue

        received = ssl_vision_wrapper.SSL_WrapperPacket()
        received.ParseFromString(message)
        if not detection and received.HasField('detection'):
            detec = MessageToDict(received.detection, preserving_proto_field_name=True)
            if detec["camera_id"] == 0:
                with open(args.gt, "w") as file:
                    yaml.dump(detec, file)
                print(f"Received and saved detection as {args.gt}")
                detection = True

        if received.HasField('geometry'):
            geom = MessageToDict(received.geometry, preserving_proto_field_name=True)
            geom["field"]["field_arcs"] = []
            geom["field"]["field_lines"] = []
            with open(args.geom, "w") as file:
                yaml.dump(geom, file)
            print(f"Received and saved detection as {args.geom}")
            geometry = True

        if detection and geometry:
            sys.exit(0)
