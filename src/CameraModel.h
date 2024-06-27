#pragma once
#include <eigen3/Eigen/Geometry>

#include "proto/ssl_vision_geometry.pb.h"

void visibleFieldExtentEstimation(const int camId, const int camAmount, const SSL_GeometryFieldSize& field, const bool withBoundary, Eigen::Vector2f &min, Eigen::Vector2f &max);


class CameraModel {
public:
	CameraModel();
	CameraModel(const Eigen::Vector2i& size, int camId, int camAmount, const SSL_GeometryFieldSize& field);

	explicit CameraModel(const SSL_GeometryCameraCalibration& calib);
	[[nodiscard]] SSL_GeometryCameraCalibration getProto(int camId) const;

	void updateDerived();
	void ensureSize(const Eigen::Vector2i& newSize);
	void updateFocalLength(float newFocalLength);

	[[nodiscard]] Eigen::Vector2f normalizeUndistort(const Eigen::Vector2f& p) const;
	[[nodiscard]] Eigen::Vector2f undistort(const Eigen::Vector2f& p) const;

	[[nodiscard]] Eigen::Vector2f field2image(const Eigen::Vector3f& p) const;
	[[nodiscard]] Eigen::Vector3f image2field(const Eigen::Vector2f& p, float height) const;

	void updateEuler(const Eigen::Vector3f& euler);
	Eigen::Vector3f getEuler();

	float focalLength;
	Eigen::Vector2f principalPoint;
	float distortionK2 = 0.0f;
	Eigen::Vector3f pos = Eigen::Vector3f(0.0f, 0.0f, 5000.0f); // camera position in field coordinate system
	Eigen::Quaternionf f2iOrientation = Eigen::Quaternionf(0.0f, -1.0f, 0.0f, 0.0f); // orientation from field to image plane
	Eigen::Vector2i size;

	Eigen::Affine3f f2iTransformation;
	Eigen::Matrix3f i2fOrientation;
};

std::ostream& operator<<(std::ostream &out, const CameraModel& data);
