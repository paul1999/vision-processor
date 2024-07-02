#!/usr/bin/env python3

#    Copyright 2024 Felix Weinmann
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
from collections import defaultdict

from binary import parser_binary, run_binary
from dataset import threaded_field_iter, parser_test_data
from record import thread_local_ip
from visionsocket import VisionRecorder

if __name__ == '__main__':
    parser = parser_test_data(parser_binary(argparse.ArgumentParser(prog='Vision blob benchmark')))
    parser.add_argument('--scenes_per_field', default=None, type=int, help='Amount of scenes per field to process')
    args = parser.parse_args()

    totalcirc = defaultdict(float)
    hitcirc = defaultdict(float)
    offset = defaultdict(float)
    size = defaultdict(int)

    def consumer(dataset):
        print(f"Recording {dataset} blob benchmark")

        recorder = VisionRecorder(vision_ip=thread_local_ip())

        def stdoutprocessor(line: str):
            print(line, end='')
            if not line.startswith("[BlobMachine]"):
                return

            split = line[:-1].split(' ')
            totalcirc[dataset] += float(split[1])
            hitcirc[dataset] += float(split[2])
            offset[dataset] += float(split[3])
            size[dataset] += 1

        for video, _ in zip(dataset.images(), range(args.scenes_per_field if args.scenes_per_field else 1000000)):
            print(f"Recording {video}")
            run_binary(args.binary, recorder, dataset, video, stdoutconsumer=stdoutprocessor)

    threaded_field_iter(args.data_folder, consumer, 1, args.field)

    for dataset, s in size.items():
        print(f"Dataset {dataset} score: Circ total {totalcirc[dataset]/s} Circ hit {hitcirc[dataset]/s} Avg offset {offset[dataset]/s}")

    totalsize = sum(size.values())
    print(f"Total score: Circ total {sum(totalcirc.values())/totalsize} Circ hit {sum(hitcirc.values())/totalsize} Avg offset {sum(offset.values())/totalsize}")
