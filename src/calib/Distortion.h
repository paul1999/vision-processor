#pragma once
#include <eigen3/Eigen/Core>
#include "CameraModel.h"

bool calibrateDistortion(const std::vector<std::vector<Eigen::Vector2f>>& linePoints, CameraModel& model);