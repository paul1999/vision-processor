#pragma once

#include <string>
#include <vector>

#include "proto/ssl_vision_detection.pb.h"


std::vector<SSL_DetectionFrame> parseGroundTruth(const std::string& source);
const SSL_DetectionFrame& getCorrespondingFrame(const std::vector<SSL_DetectionFrame>& groundTruth, uint32_t frameId);

