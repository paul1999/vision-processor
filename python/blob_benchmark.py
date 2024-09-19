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

    blobs = defaultdict(int)
    errorSum = defaultdict(float)
    sqErrorSum = defaultdict(float)
    fieldScale = defaultdict(float)

    frames = defaultdict(int)
    worstBlobSum = defaultdict(float)
    percentileSum = defaultdict(float)

    balls = defaultdict(int)
    ballErrorSum = defaultdict(float)
    ballSqErrorSum = defaultdict(float)

    bots = defaultdict(int)
    botErrorSum = defaultdict(float)
    botSqErrorSum = defaultdict(float)

    def consumer(dataset):
        print(f"Recording {dataset} blob benchmark")

        recorder = VisionRecorder(vision_ip=thread_local_ip())

        def stdoutprocessor(line: str):
            if not line.startswith("[BlobMachine]"):
                return

            key = dataset.folder.parent.name
            split = line[:-1].split(' ')
            frames[key] += int(split[1])
            blobs[key] += int(split[2])
            errorSum[key] += float(split[3])
            sqErrorSum[key] += float(split[4])
            worstBlobSum[key] += float(split[5])
            percentileSum[key] += float(split[6])
            balls[key] += int(split[7])
            ballErrorSum[key] += float(split[8])
            ballSqErrorSum[key] += float(split[9])
            bots[key] += int(split[10])
            botErrorSum[key] += float(split[11])
            botSqErrorSum[key] += float(split[12])
            fieldScale[key] += float(split[13])
            #if float(split[5]) < 0:
            #    print(f"  NEGATIVE WBS")

        for video, _ in zip(dataset.images(), range(args.scenes_per_field if args.scenes_per_field else 1000000)):
            print(f"Processing {video}")
            run_binary(args.binary, recorder, dataset, video, stdoutconsumer=stdoutprocessor)  # , ground_truth=video.with_suffix('.vision_processor.json')

    threaded_field_iter(args.data_folder, consumer, field_filter=args.field)

    def errorStddev(error, sqError, amount):
        return error / amount, math.sqrt(amount * sqError - error ** 2) / amount

    totalError = 0.0
    totalStddev = 0.0
    totalBallError = 0.0
    totalBallStddev = 0.0
    totalBotError = 0.0
    totalBotStddev = 0.0
    totalPsr = 0.0
    totalEfsr = 0.0
    for dataset, b in blobs.items():
        error, stddev = errorStddev(errorSum[dataset], sqErrorSum[dataset], b)
        ballError, ballStddev = errorStddev(ballErrorSum[dataset], ballSqErrorSum[dataset], balls[dataset])
        botError, botStddev = errorStddev(botErrorSum[dataset], botSqErrorSum[dataset], bots[dataset])
        psr = worstBlobSum[dataset] / abs(worstBlobSum[dataset] + percentileSum[dataset])
        efsr = errorSum[dataset] / fieldScale[dataset]
        print(f"  {dataset: >11} blobs: {error: .2f}±{stddev: .2f} balls: {ballError: .2f}±{ballStddev: .2f} bots: {botError: .2f}±{botStddev: .2f} PSR {psr: .4f} EFSR {efsr}")

        totalError += error
        totalStddev += stddev
        totalBallError += ballError
        totalBallStddev += ballStddev
        totalBotError += botError
        totalBotStddev += botStddev
        totalPsr += psr
        totalEfsr += efsr

    d = len(blobs.keys())
    print(f"Total blobs: {totalError/d: .2f}±{totalStddev/d: .2f} balls: {totalBallError/d: .2f}±{totalBallStddev/d: .2f} bots: {totalBotError/d: .2f}±{totalBotStddev/d: .2f} PSR {totalPsr/d: .4f} EFSR {totalEfsr/d: .4f}")
