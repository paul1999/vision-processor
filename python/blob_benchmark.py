#!/usr/bin/env python3
import argparse

from binary import parser_binary, run_binary
from dataset import threaded_field_iter, parser_test_data
from record import thread_local_ip
from visionsocket import VisionRecorder

if __name__ == '__main__':
    args = parser_test_data(parser_binary(argparse.ArgumentParser(prog='Vision blob benchmark'))).parse_args()

    def consumer(dataset):
        print(f"Recording {dataset} geometry")

        recorder = VisionRecorder(vision_ip=thread_local_ip())

        # TODO record score
        for video in dataset.images():
            print(f"Recording {video}")
            run_binary(args.binary, recorder, dataset, dataset.field, stdoutconsumer=lambda line: print(line, end=''))

    threaded_field_iter(args.data_folder, consumer, 1)

