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
import math
from collections import defaultdict

from binary import parser_binary, run_binary
from dataset import threaded_field_iter, parser_test_data
from record import thread_local_ip
from visionsocket import VisionRecorder

if __name__ == '__main__':
    parser = parser_test_data(parser_binary(argparse.ArgumentParser(prog='Vision blob benchmark'), default='blob_benchmark'))
    parser.add_argument('--scenes_per_field', default=None, type=int, help='Amount of scenes per field to process')
    args = parser.parse_args()

    frames = defaultdict(int)
    blobs = defaultdict(int)
    errorSum = defaultdict(float)
    sqErrorSum = defaultdict(float)
    worstBlobSum = defaultdict(float)
    percentileSum = defaultdict(float)

    def consumer(dataset):
        print(f"Recording {dataset} blob benchmark")

        recorder = VisionRecorder(vision_ip=thread_local_ip())

        def stdoutprocessor(line: str):
            #print(line, end='')
            if not line.startswith("[BlobMachine]"):
                return

            key = dataset.folder.parent
            split = line[:-1].split(' ')
            frames[key] += int(split[1])
            blobs[key] += int(split[2])
            errorSum[key] += float(split[3])
            sqErrorSum[key] += float(split[4])
            worstBlobSum[key] += float(split[5])
            percentileSum[key] += float(split[6])

        for video, _ in zip(dataset.images(), range(args.scenes_per_field if args.scenes_per_field else 1000000)):
            print(f"Processing {video}")
            run_binary(args.binary, recorder, dataset, video, stdoutconsumer=stdoutprocessor)

    threaded_field_iter(args.data_folder, consumer, field_filter=args.field)

    totalError = 0.0
    totalStddev = 0.0
    totalPsr = 0.0
    for dataset, b in blobs.items():
        error = errorSum[dataset] / b
        stddev = math.sqrt(b*sqErrorSum[dataset] - errorSum[dataset]**2) / b
        psr = worstBlobSum[dataset] / abs(worstBlobSum[dataset] + percentileSum[dataset])
        print(f"  {dataset} error: {error: .2f}±{stddev: .2f} PSR {psr: .4f}")

        totalError += error
        totalStddev += stddev
        totalPsr += psr

    datasets = len(blobs.keys())
    print(f"Total error: {totalError/datasets: .2f}±{totalStddev/datasets: .2f} PSR {totalPsr/datasets: .4f}")
