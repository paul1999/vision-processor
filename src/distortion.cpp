#include "distortion.h"

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

	//float e = 0;
	std::vector<float> error;
	for(const Eigen::Vector2f& u : undistorted) {
		float pe = pointError(n, u, d0);
		if(isnanf(pe)) {
			std::cout << "NaNP: " << std::endl << n << std::endl << u << std::endl << d0 << " " << Ex << " " << Ey << std::endl;
			exit(1);
		}
		error.push_back(pe);
		//e += pe*pe;
	}
	//return e;
	return error;
}

/*float modelError(const std::vector<std::vector<Eigen::Vector2f>>& lines, float k3, float k5) {
	float e = 0;

	for(const std::vector<Eigen::Vector2f>& distorted : lines) {
		std::vector<Eigen::Vector2f> undistorted;
		for(const Eigen::Vector2f& d : distorted) {
			float r2 = d.x()*d.x() + d.y()*d.y();
			float factor = 1 + k3*r2 + k5*r2*r2;
			undistorted.emplace_back(d.x()*factor, d.y()*factor);
		}

		e += lineError(undistorted);
	}

	return e;
}*/

struct Functor : public Eigen::DenseFunctor<float> {
	const std::vector<std::vector<Eigen::Vector2f>>& lines;

	//TODO swap inputs and values?
	explicit Functor(const std::vector<std::vector<Eigen::Vector2f>>& lines): lines(lines) {} // Eigen::DenseFunctor<float>(2, lines.size()),

	int operator()(const InputType &x, ValueType& fvec) const {
		int i = 0;
		//for(int i = 0; i < lines.size(); i++) {
		for(const std::vector<Eigen::Vector2f>& distorted : lines) {
			//const std::vector<Eigen::Vector2f>& distorted = lines[i];
			std::vector<Eigen::Vector2f> undistorted;
			for(const Eigen::Vector2f& d : distorted) {
				float r2 = d.x()*d.x() + d.y()*d.y();
				float factor = 1 + x(0)*r2 + x(1)*r2*r2;
				undistorted.emplace_back(d.x()*factor, d.y()*factor);
			}

			std::vector<float> error = lineError(undistorted);
			for(float e : error)
				fvec(i++) = e;
			//fvec(i) = lineError(undistorted);
		}
		return 0;
	}

	int values() const {
		int size = 0;
		for (const auto& item : lines)
			size += item.size();
		return size;
	}

	int inputs() const {
		return 2;
	}
};

Eigen::Vector2f distortion(const std::vector<std::vector<Eigen::Vector2f>>& lines) {
	//Eigen::NumericalDiff<std::functor<Eigen::Vector2f>>
	Functor functor(lines);
	Eigen::NumericalDiff<Functor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Functor>> lm(numDiff);
	//lm.setFactor(0.01f);

	//TODO for each point/line
	Eigen::VectorXf k(2);
	k(0) = 0;
	k(1) = 0;
	int ret = lm.minimize(k);
	std::cout << lm.iterations() << " " << ret << " " << k(0) << " " << k(1) << std::endl;

	return {k(0), k(1)};
}