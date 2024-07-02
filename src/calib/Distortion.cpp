/*
     Copyright 2024 Felix Weinmann

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
 */
#include "Distortion.h"

#include <eigen3/unsupported/Eigen/LevenbergMarquardt>
#include <iostream>

static inline float pointError(const Eigen::Vector2f& n, const Eigen::Vector2f& u, const float d0) {
	return n.transpose().dot(u) - d0;
}

static std::vector<float> lineError(const std::vector<Eigen::Vector2f>& undistorted) {
	float Ex = 0;
	float Ey = 0;
	float Exx = 0;
	float Eyy = 0;
	float Exy = 0;
	for(const Eigen::Vector2f& u : undistorted) {
		Ex += u.x();
		Ey += u.y();
		Exx += u.x()*u.x();
		Eyy += u.y()*u.y();
		Exy += u.x()*u.y();
	}
	Ex /= undistorted.size();
	Ey /= undistorted.size();
	Exx /= undistorted.size();
	Eyy /= undistorted.size();
	Exy /= undistorted.size();

	Eigen::Vector2f n;
	float d0;
	if (Exx - Ex*Ex >= Eyy - Ey*Ey) {
		float a = (Exy - Ex*Ey) / (Exx - Ex*Ex);
		float b = (Exx*Ey - Ex*Exy) / (Exx - Ex*Ex);
		n.x() = -a / sqrtf(a*a + 1);
		n.y() = 1 / sqrtf(a*a + 1);
		d0 = b / sqrtf(a*a + 1);
	} else {
		float c = (Exy - Ex*Ey) / (Eyy - Ey*Ey);
		float d = (Eyy*Ex - Ey*Exy) / (Eyy - Ey*Ey);
		n.x() = 1 / sqrtf(c*c + 1);
		n.y() = -c / sqrtf(c*c + 1);
		d0 = d / sqrtf(c*c + 1);
	}

	std::vector<float> error;
	for(const Eigen::Vector2f& u : undistorted) {
		float pe = pointError(n, u, d0);
		if(isnanf(pe)) {
			std::cout << "NaNP: " << std::endl << n << std::endl << u << std::endl << d0 << " " << Ex << " " << Ey << std::endl;
			exit(1);
		}
		error.push_back(pe);
	}
	return error;
}

struct Functor : public Eigen::DenseFunctor<float> {
	const std::vector<std::vector<Eigen::Vector2f>>& lines;
	CameraModel& reference;

	explicit Functor(const std::vector<std::vector<Eigen::Vector2f>>& lines, CameraModel& model): lines(lines), reference(model) {}

	int operator()(const InputType &x, ValueType& fvec) const {
		CameraModel model = reference;
		model.distortionK2 = x(0);
		model.principalPoint.x() = x(1);
		model.principalPoint.y() = x(2);

		int i = 0;
		for(const std::vector<Eigen::Vector2f>& distorted : lines) {
			std::vector<Eigen::Vector2f> undistorted;
			for(const Eigen::Vector2f& d : distorted)
				undistorted.push_back(model.normalizeUndistort(d));

			std::vector<float> error = lineError(undistorted);
			for(float e : error)
				fvec(i++) = e;
		}
		return 0;
	}

	int values() const {
		int size = 0;
		for (const auto& item : lines)
			size += item.size();
		return size;
	}
};

bool calibrateDistortion(const std::vector<std::vector<Eigen::Vector2f>>& linePoints, CameraModel& model) {
	Functor functor(linePoints, model);
	std::cout << "[Distortion] Lines: " << linePoints.size() << " with line points: " << functor.values() << std::endl;
	Eigen::NumericalDiff<Functor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Functor>> lm(numDiff);
	Eigen::VectorXf k(3);
	k(0) = model.distortionK2;
	k(1) = model.principalPoint.x();
	k(2) = model.principalPoint.y();

	std::cout << lm.minimize(k) << " " << lm.iterations() << std::endl;

	if(lm.info() != Eigen::ComputationInfo::Success) {
		std::cout << "[Distortion] Levenberg-Marquandt minimization failed with code, aborting calibration for this frame: " << lm.info() << std::endl;
		return false;
	}

	std::cout << "[Distortion] Determined parameters: distortion " << k(0) << " principal point " << k(1) << "|" << k(2) << std::endl;
	if(k(1) < 0.0f || k(2) < 0.0f || k(1) >= model.size.x() || k(2) >= model.size.y()) {
		std::cout << "[Distortion] Principal point outside of image, aborting calibration for this frame" << std::endl;
		return false;
	}

	model.distortionK2 = k(0);
	model.principalPoint.x() = k(1);
	model.principalPoint.y() = k(2);
	return true;
}