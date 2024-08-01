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
from pathlib import Path

import yaml

from dataset import parser_test_data, threaded_field_iter


if __name__ == '__main__':
    args = parser_test_data(argparse.ArgumentParser(prog='Vision recorder')).parse_args()

    # [binary][dataset][video][type]
    frames = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
    detection_rates = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))

    def consumer(dataset):
        global frames, detection_rates

        for video in dataset.images():
            video: Path = video
            print(f"Processing {video}")

            detections = {}
            for records in video.parent.glob(video.stem + '.*.yml'):
                binary = records.stem.split('.')[-1]
                with records.open('r') as file:
                    detections[binary] = yaml.load(file, yaml.CBaseLoader)

            local_detection_rates = defaultdict(lambda: defaultdict(lambda: 0))
            for binary, detection_list in detections.items():
                for frame in detection_list:
                    if 'balls' in frame and len(frame['balls']) == 1:
                        local_detection_rates[binary]['ball'] += 1
                    if 'robots_yellow' in frame:
                        for bot in frame['robots_yellow']:
                            local_detection_rates[binary]['y' + str(bot['robot_id'])] += 1
                    if 'robots_blue' in frame:
                        for bot in frame['robots_blue']:
                            local_detection_rates[binary]['b' + str(bot['robot_id'])] += 1

            video_frames = max(len(detection_list) for detection_list in detections.values())
            objects = {t for binary in local_detection_rates.keys() for t in local_detection_rates[binary].keys()}
            correct_objects = {t for t in objects if max(detection_rate[t] for detection_rate in local_detection_rates.values()) / video_frames >= 0.2}  # At least 20% occurance from one of the binaries
            for binary in local_detection_rates.keys():
                for t in correct_objects:
                    detection_rates[binary][dataset][video][t] = local_detection_rates[binary][t]
                    frames[binary][dataset][video][t] = video_frames
                #for t in objects - correct_objects:
                #    detection_rates[binary][dataset][video]['false'] += local_detection_rates[binary][t]
                #    frames[binary][dataset][video]['false'] = video_frames

            #TODO binary detection offset

    try:
        threaded_field_iter(args.data_folder, consumer, field_filter=args.field)
    except KeyboardInterrupt:
        pass

    def dictsum(d: dict, binary_filter=None, dataset_filter=None, video_filter=None, object_filter=None) -> int:
        return sum(
            amount
            for binary, d1 in d.items() if binary_filter is None or binary == binary_filter
            for dataset, d2 in d1.items() if dataset_filter is None or dataset == dataset_filter
            for video, d3 in d2.items() if video_filter is None or video == video_filter
            for object, amount in d3.items() if object_filter is None or object == object_filter
        )

    for binary in detection_rates.keys():
        print(f"--- {binary} ---")
        print(f"Total {dictsum(detection_rates, binary_filter=binary) / dictsum(frames, binary_filter=binary)}")

        min_rate = None, 1.0
        for dataset in detection_rates[binary].keys():
            rate = dictsum(detection_rates, binary_filter=binary, dataset_filter=dataset) / dictsum(frames, binary_filter=binary, dataset_filter=dataset)
            if rate < min_rate[1]:
                min_rate = dataset, rate
        print(f"Worst dataset {min_rate[0]} {min_rate[1]}")

        min_video = None, None, 1.0
        for dataset in detection_rates[binary].keys():
            for video in detection_rates[binary][dataset].keys():
                rate = dictsum(detection_rates, binary_filter=binary, dataset_filter=dataset, video_filter=video) / dictsum(frames, binary_filter=binary, dataset_filter=dataset, video_filter=video)
                if rate < min_video[2]:
                    min_video = dataset, video, rate
        print(f"Worst video {min_video[0]} {min_video[1]} {min_video[2]}")

        min_type = None, 1.0
        for t in {t for d1 in detection_rates[binary].values() for d2 in d1.values() for t in d2.keys()}:
            if t != 'false':
                rate = dictsum(detection_rates, binary_filter=binary, object_filter=t) / dictsum(frames, binary_filter=binary, object_filter=t)
                if rate < min_type[1]:
                    min_type = t, rate
        print(f"Worst type {min_type[0]} {min_type[1]}")

        print()

