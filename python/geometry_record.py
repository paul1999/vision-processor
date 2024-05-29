#!/usr/bin/env python3
import argparse
import sys

import yaml
from google.protobuf.json_format import MessageToDict

from binary import parser_binary, run_binary
from dataset import threaded_field_iter, parser_test_data
from record import thread_local_ip
from visionsocket import VisionRecorder

if __name__ == '__main__':
    args = parser_test_data(parser_binary(argparse.ArgumentParser(prog='Vision geometry recorder'))).parse_args()

    def consumer(dataset):
        print(f"Recording {dataset} geometry")

        recorder = VisionRecorder(vision_ip=thread_local_ip())

        reference_geometry = dataset.reference_geometry
        del reference_geometry.geometry.calib[:]

        # TODO record score
        run_binary(args.binary, recorder, dataset, dataset.field, reference_geometry)

        geometries = recorder.subfield('geometry')

        if len(geometries[-1].calib) > 0:
            with (dataset.folder / ('geometry.' + args.binary.name + '.yml')).open('w') as file:
                yaml.dump(MessageToDict(geometries[-1], preserving_proto_field_name=True), file)
        else:
            print(f"No calibration received!", file=sys.stderr)

    threaded_field_iter(args.data_folder, consumer, 1)

