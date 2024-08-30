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
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from dataset import parser_test_data, threaded_field_iter, Dataset

if __name__ == '__main__':
    args = parser_test_data(argparse.ArgumentParser(prog='Vision recorder')).parse_args()

    # [binary][dataset][cam][video][type]
    frames = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))))
    truepositive_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))))

    def confidence(object):
        confidence = object['confidence']
        if type(confidence) is float:
            return confidence
        else:
            return float(confidence.replace('.nan', '1.0'))

    def consumer(dataset: Dataset):
        global frames, truepositive_rates

        for video in dataset.images():
            video: Path = video
            print(f"Processing {video}")

            detections = {}
            for records in video.parent.glob(video.stem + '.*.json'):
                binary = records.stem.split('.')[-1]
                with records.open('r') as file:
                    detections[binary] = json.load(file)

            with_manual = 'manual' in detections.keys()
            video_frames = max(len(detection_list) for detection_list in detections.values())

            local_detection_rates = defaultdict(lambda: defaultdict(lambda: 0))
            for binary, detection_list in detections.items():
                binary_detection_rates = local_detection_rates[binary]
                for frame in detection_list:
                    if 'balls' in frame and (with_manual or (len(frame['balls']) == 1 and confidence(frame['balls'][0]) > 0.1)):
                        binary_detection_rates['ball'] += len(frame['balls'])
                    if 'robots_yellow' in frame:
                        for bot in frame['robots_yellow']:
                            if confidence(bot) > 0.1:
                                binary_detection_rates['y' + str(bot['robot_id'])] += 1
                    if 'robots_blue' in frame:
                        for bot in frame['robots_blue']:
                            if confidence(bot) > 0.1:
                                binary_detection_rates['b' + str(bot['robot_id'])] += 1

            objects = {t for binary in local_detection_rates.keys() for t in local_detection_rates[binary].keys()}
            if with_manual:
                correct_objects = {t for t in local_detection_rates['manual'].keys()}
            else:
                correct_objects = {t for t in objects if max(detection_rate[t] for detection_rate in local_detection_rates.values()) / video_frames >= 0.2}  # At least 20% occurance from one of the binaries
            for binary in local_detection_rates.keys():
                for t in correct_objects:
                    if with_manual:
                        reference = local_detection_rates['manual'][t]
                        # Detect and punish too many detections.
                        truepositive_rates[binary][dataset.folder.parent][dataset.folder.name][video][t] = max(min(local_detection_rates[binary][t], 2 * reference - local_detection_rates[binary][t]), 0)
                        frames[binary][dataset.folder.parent][dataset.folder.name][video][t] = reference
                    else:
                        truepositive_rates[binary][dataset.folder.parent][dataset.folder.name][video][t] = local_detection_rates[binary][t]
                        frames[binary][dataset.folder.parent][dataset.folder.name][video][t] = video_frames
                #for t in objects - correct_objects:
                #    detection_rates[binary][dataset][video]['false'] += local_detection_rates[binary][t]
                #    frames[binary][dataset][video]['false'] = video_frames

            #TODO binary detection offset

    try:
        threaded_field_iter(args.data_folder, consumer, 1, field_filter=args.field)
    except KeyboardInterrupt:
        pass

    def dsum(d: dict, generator=lambda x: x, filter=None) -> float:
        return sum(generator(value) for key, value in d.items() if filter is None or filter == key)

    def dictmean(d: dict, s: dict, dgenerator=lambda x, y: x / y, filter=None) -> float:
        return fmean(
            value
            for value in (
                dgenerator(d[key], value)
                for key, value in s.items()
                if filter is None or filter == key
            )
            if value is not math.nan
        )

    def detection_rate(binary, dataset_filter=None, video_filter=None, object_filter=None):
        def camsum(cams):
            return dsum(
                cams,
                lambda videos: dsum(
                    videos,
                    lambda objects: dsum(
                        objects,
                        filter=object_filter
                    ),
                    video_filter
                )
            )

        def cammean(x, y):
            try:
                return camsum(x) / camsum(y)
            except ZeroDivisionError:
                return math.nan


        return dictmean(
            truepositive_rates[binary], frames[binary],
            cammean,
            dataset_filter
        )

    for binary in truepositive_rates.keys():
        print(f"--- {binary} ---")
        print(f"Total {detection_rate(binary)}")

        min_rate = None, 1.0
        img_rate = []
        video_rate = []
        for dataset in truepositive_rates[binary].keys():
            rate = detection_rate(binary, dataset)
            if len(list(dataset.glob('*/*.mp4'))) == 0:
                img_rate.append(rate)
            else:
                video_rate.append(rate)

            print(f"  Dataset {dataset.name} {rate}")
            if rate < min_rate[1]:
                min_rate = dataset, rate

        if video_rate:
            print(f"Video {fmean(video_rate)}")
        if img_rate:
            print(f"Image {fmean(img_rate)}")

        print(f"Worst dataset {min_rate[0]} {min_rate[1]}")

        min_video = None, 1.0
        for dataset in truepositive_rates[binary].keys():
            for camera in truepositive_rates[binary][dataset].keys():
                for video in truepositive_rates[binary][dataset][camera].keys():
                    rate = detection_rate(binary, dataset, video)
                    if rate < min_video[1]:
                        min_video = video, rate
        print(f"Worst video {min_video[0]} {min_video[1]}")

        # TODO unweighted average?
        min_type = None, 1.0
        for t in {t for cams in truepositive_rates[binary].values() for videos in cams.values() for types in videos.values() for t in types.keys()}:
            if t != 'false':
                rate = detection_rate(binary, object_filter=t)
                if rate < min_type[1]:
                    min_type = t, rate
        print(f"Worst type {min_type[0]} {min_type[1]}")

        print()

