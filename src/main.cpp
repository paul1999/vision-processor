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
#include <csignal>
#include <iostream>
#include <iomanip>
#include <opencv2/bgsegm.hpp>
#include <yaml-cpp/yaml.h>

#include "proto/ssl_vision_wrapper.pb.h"
#include "Resources.h"
#include "GroundTruth.h"
#include "calib/GeomModel.h"
#include "pattern.h"
#include "cl_kernels.h"
#include <opencv2/video/background_segm.hpp>
#include "KDTree.h"

struct __attribute__ ((packed)) CLMatch {
	float x, y;
	RGB color;
	RGB center;
	float circ;
	float score;

	auto operator<=>(const CLMatch&) const = default;
};

struct Match {
	Eigen::Vector2f pos;
	Eigen::Vector3i color;
	Eigen::Vector3i center;
	float circ;
	float score;

	auto operator<=>(const Match&) const = default;
};

static bool kMeans(const Eigen::Vector3i& different, const std::vector<Eigen::Vector3i>& values, Eigen::Vector3i& c1, Eigen::Vector3i& c2) {
	if(values.size() < 2)
		return false;

	float inGroupDiff = INFINITY;
	float outGroupDiff = INFINITY;

	for (int i = 0; i < values.size(); i++) {
		const auto& value = values[i];
		outGroupDiff = std::min(outGroupDiff, (float)(value - different).squaredNorm());

		for (int j = i+1; j < values.size(); j++) {
			inGroupDiff = std::min(inGroupDiff, (float)(values[j] - value).squaredNorm());
		}
	}

	if(inGroupDiff > outGroupDiff) {
		//std::cerr << "   Ingroup bigger than outgroup" << std::endl;
		return false;
	}

	inGroupDiff = sqrtf(inGroupDiff);
	outGroupDiff = sqrtf(outGroupDiff);

	Eigen::Vector3i c1backup = c1;
	Eigen::Vector3i c2backup = c2;

	//https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
	//https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
	c1 = *std::min_element(values.begin(), values.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) { return (a - c1).squaredNorm() < (b - c1).squaredNorm(); });
	c2 = *std::min_element(values.begin(), values.end(), [&](const Eigen::Vector3i& a, const Eigen::Vector3i& b) { return (a - c2).squaredNorm() < (b - c2).squaredNorm(); });
	if(c1 == c2) {
		c1 = c1backup;
		c2 = c2backup;
		return false;
	}

	Eigen::Vector3i oldC1 = c2;
	Eigen::Vector3i oldC2 = c1;
	int n1 = 0;
	int n2 = 0;
	while(oldC1 != c1 && oldC2 != c2) {
		Eigen::Vector3i sum1 = {0, 0, 0};
		Eigen::Vector3i sum2 = {0, 0, 0};
		n1 = 0;
		n2 = 0;
		for (const auto& value : values) {
			if((value - c1).squaredNorm() < (value - c2).squaredNorm()) {
				sum1 += value;
				n1++;
			} else {
				sum2 += value;
				n2++;
			}
		}

		if(n1 == 0 || n2 == 0) {
			//std::cerr << "   N0 " << n1 << "|" << n2 << "   " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
			c1 = c1backup;
			c2 = c2backup;
			return false;
		}

		oldC1 = c1;
		oldC2 = c2;
		c1 = sum1 / n1;
		c2 = sum2 / n2;
	}

	float mergeRange = (outGroupDiff - inGroupDiff) / 2.0f;
	if((float)(c1 - c2).norm() < mergeRange) {
		//std::cerr << "   Skipping Update for " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
		c1 = c1backup;
		c2 = c2backup;
		return false;
	}

	/*if((c1 - c2).dot(c1backup - c2backup) <= 0) { //TODO did never trigger
		std::cerr << "   Attempted color direction inversion" << std::endl;
		c1 = c1backup;
		c2 = c2backup;
	}*/

	// https://en.wikipedia.org/wiki/Silhouette_(clustering)#Simplified_Silhouette_and_Medoid_Silhouette
	/*float s1 = 0.0;
	float s2 = 0.0;
	for (const auto& value : values) {
		float a = (float)(value - c1).norm();
		float b = (float)(value - c2).norm();
		if(a < b) {
			s1 += (b - a) / b;
		} else {
			s2 += (a - b) / a;
		}
	}

	//TODO circularity of samples: if roughly circular: one cluster?
	//TODO if small sample size: combine multiple frames
	if(std::max(s1/(float)n1, s2/(float)n2) < 1.0f) { //TODO not working	 //TODO hardcoded value
		std::cerr << "   Skipping Update for " << n1 << "|" << n2 << "   " << c1backup.transpose() << "|" << c2backup.transpose() << "   " << c1.transpose() << "|" << c2.transpose() << std::endl;
		c1 = c1backup;
		c2 = c2backup;
	}*/

	return true;
}


class KDTree {
public:
	KDTree(): size(0), data(nullptr) {}
	KDTree(Match* data): data(data) {}
	KDTree(int dim, Match* data): dim(dim), data(data) {}

	void insert(Match* iData) {
		size++;
		std::unique_ptr<KDTree>& side = iData->pos[dim] < data->pos[dim] ? left : right;
		if(side != nullptr) {
			side->insert(iData);
			return;
		}

		side = std::make_unique<KDTree>((dim+1) % 2, iData);
	}

	void rangeSearch(std::vector<Match*>& values, const Eigen::Vector2f& point, const float radius) {
		if((data->pos - point).norm() <= radius)
			values.push_back(data);

		if(left != nullptr && point[dim] <= data->pos[dim] + radius)
			left->rangeSearch(values, point, radius);
		if(right != nullptr && point[dim] >= data->pos[dim] - radius)
			right->rangeSearch(values, point, radius);
	}

	inline int getSize() const { return size; }

	std::unique_ptr<KDTree> left = nullptr;
	std::unique_ptr<KDTree> right = nullptr;
	Match* data;
private:
	int dim = 0;
	int size = 1;
};

class BallHypothesis {
public:
	BallHypothesis(const Resources& r, const Match* blob): blob(blob), pos(blob->pos) {
		calcColorScore(r);
	}

	void recalcPostColorCalib(const Resources& r) {
		score = 1.0f;
		calcColorScore(r);
	}

	void addToDetectionFrame(const Resources& r, SSL_DetectionFrame* detection) {
		const Eigen::Vector2f imgPos = r.perspective->model.field2image({pos.x(), pos.y(), (float)r.gcSocket->maxBotHeight});
		const Eigen::Vector3f ballPos = r.perspective->model.image2field(imgPos, r.perspective->field.ball_radius());
		SSL_DetectionBall* ball = detection->add_balls();
		ball->set_confidence(score);
		//ball->set_area(0);
		ball->set_x(ballPos.x());
		ball->set_y(ballPos.y());
		//ball->set_z(0.0f);
		ball->set_pixel_x(imgPos.x());
		ball->set_pixel_y(imgPos.y());
	}

	const Match* blob;
	Eigen::Vector2f pos = {0, 0};
	float score = 1.0f;

private:
	void calcColorScore(const Resources& r) {
		//TODO test center vs color (updateColors as well)
		int falseOrange = (blob->center - r.falseOrange).squaredNorm();
		int orange = (blob->center - r.orange).squaredNorm();

		if (falseOrange <= orange) {
			score = 0;
			return;
		}

		score *= 1 - (float)orange / (float)falseOrange;
	}
};

class BotHypothesis {
public:
	BotHypothesis(const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): blobs{a, b, c, d, e} {
		for(auto& blob : blobs)
			if(blob != nullptr)
				blobAmount++;

		calcPos();
		calcOffsetScore();
	}

	[[nodiscard]] bool isClipping(const BotHypothesis& other) const {
		Eigen::Vector2f diff = other.pos - pos;
		float sqDistance = diff.squaredNorm();
		if(sqDistance >= (2*MIN_ROBOT_RADIUS)*(2*MIN_ROBOT_RADIUS)) //Early rejection for faster calculation (simple circle - circle clipping)
			return false;

		float diffAngle = atan2f(diff.y(), diff.x());
		float selfAngle = remainderf(diffAngle - orientation, 2.0f * M_PI);
		float otherAngle = remainderf(diffAngle - other.orientation, 2.0f * M_PI);

		float minDistance =
				(abs(selfAngle) < MIN_ROBOT_OPENING_ANGLE ? MIN_ROBOT_FRONT_DISTANCE/cosf(selfAngle) : MIN_ROBOT_RADIUS) +
				(abs(otherAngle) < MIN_ROBOT_OPENING_ANGLE ? MIN_ROBOT_FRONT_DISTANCE/cosf(otherAngle) : MIN_ROBOT_RADIUS);

		return sqDistance < minDistance*minDistance;
	}

	[[nodiscard]] bool isClipping(const Resources& r, const BallHypothesis& ball) const {
		const float clippedBallRadius = 0.48837 * r.perspective->field.ball_radius(); // a ball may clip up to 20% of the top-view area into the robot
		Eigen::Vector2f diff = ball.pos - pos;
		float sqDistance = diff.squaredNorm();
		float minDistance = MIN_ROBOT_RADIUS + clippedBallRadius;
		if(sqDistance >= minDistance*minDistance)
			return false;

		float angle = remainderf(atan2f(diff.y(), diff.x()) - orientation, 2.0f * M_PI);
		if(abs(angle) >= MIN_ROBOT_OPENING_ANGLE)
			return true;

		minDistance = (MIN_ROBOT_FRONT_DISTANCE + clippedBallRadius) / cosf(angle);

		return sqDistance < minDistance*minDistance;
	}

	void addToDetectionFrame(const Resources& r, SSL_DetectionFrame* detection) {
		bool yellow = botId < 16;
		const Eigen::Vector2f imgPos = r.perspective->model.field2image({pos.x(), pos.y(), (float)r.gcSocket->maxBotHeight});
		const Eigen::Vector3f botPos = r.perspective->model.image2field(imgPos, (float)(yellow ? r.gcSocket->yellowBotHeight : r.gcSocket->blueBotHeight));
		SSL_DetectionRobot* bot = yellow ? detection->add_robots_yellow() : detection->add_robots_blue();
		bot->set_confidence(score);
		bot->set_robot_id(botId % 16);
		bot->set_x(botPos.x());
		bot->set_y(botPos.y());
		bot->set_height(botPos.z());
		bot->set_orientation(orientation);
		bot->set_pixel_x(imgPos.x());
		bot->set_pixel_y(imgPos.y());
	}

	virtual void recalcPostColorCalib(const Resources& r) = 0;

	const Match* blobs[5];
	Eigen::Vector2f pos = {0, 0};
	float orientation = 0;
	float score = 1.0f;
	float offsetScore = 1.0f;
	int botId = -1;
	int blobAmount = 0;

private:
	inline void calcPos() {
		//https://www.themathdoctors.org/averaging-angles/
		float oSin = 0;
		float oCos = 0;
		for(int a = 0; a < 5; a++) {
			if(blobs[a] == nullptr)
				continue;

			for(int b = a+1; b < 5; b++) {
				if(blobs[b] == nullptr)
					continue;

				Eigen::Vector2f diff = blobs[b]->pos - blobs[a]->pos;
				float angleDelta = atan2f(diff.y(), diff.x()) - patternAnglesb2b[b*5 + a];
				oSin += sinf(angleDelta);
				oCos += cosf(angleDelta);
			}
		}
		if(blobAmount < 1)
			return;

		if(blobAmount > 1)
			orientation = atan2f(oSin, oCos);

		Eigen::Rotation2Df rotation(orientation);
		pos.x() = 0;
		pos.y() = 0;
		for(int i = 0; i < 5; i++) {
			if(blobs[i] == nullptr)
				continue;

			pos += blobs[i]->pos - rotation * patternPos[i];
		}

		pos /= (float)blobAmount;
	}

	inline void calcOffsetScore() {
		Eigen::Rotation2Df rotation(orientation);
		for(int i = 0; i < 5; i++) {
			const Match* const& blob = blobs[i];
			if(blob == nullptr)
				continue;

			Eigen::Vector2f offset = (blob->pos - (pos + rotation * patternPos[i])) / 10.0f; // (10.0f) 1cm offset -> 0.5 score
			offsetScore = std::min(offsetScore, 1 / (1 + offset.squaredNorm()));
		}

		score = offsetScore;
	}
};

class DetectionBotHypothesis: public BotHypothesis {
public:
	DetectionBotHypothesis(const Resources& r, const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): BotHypothesis(a, b, c, d, e) {
		calcBotId(r);
	}

	void recalcPostColorCalib(const Resources &r) override {
		calcBotId(r);
	}

private:
	inline void calcBotId(const Resources& r) {
		Eigen::Vector3i green = r.green;
		Eigen::Vector3i pink = r.pink;
		kMeans(blobs[0]->color, {blobs[1]->color, blobs[2]->color, blobs[3]->color, blobs[4]->color}, green, pink);

		botId = ((blobs[0]->color - r.blue).squaredNorm() < (blobs[0]->color - r.yellow).squaredNorm() ? 16 : 0) + patternLUT[
				(((blobs[1]->color - green).squaredNorm() < (blobs[1]->color - pink).squaredNorm() ? 1 : 0) << 3) +
				(((blobs[2]->color - green).squaredNorm() < (blobs[2]->color - pink).squaredNorm()? 1 : 0) << 2) +
				(((blobs[3]->color - green).squaredNorm() < (blobs[3]->color - pink).squaredNorm() ? 1 : 0) << 1) +
				((blobs[4]->color - green).squaredNorm() < (blobs[4]->color - pink).squaredNorm() ? 1 : 0)
		];
	}
};

class TrackedBotHypothesis: public BotHypothesis {
public:
	TrackedBotHypothesis(const Resources& r, const TrackingState& tracked, const Eigen::Vector3f& trackedPosition, const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): BotHypothesis(a, b, c, d, e), trackedScore(tracked.confidence), trackedPosition(trackedPosition) {
		botId = tracked.id;

		float rotationOffset = remainderf(orientation - trackedPosition.z(), 2.0f * M_PI) / (float)M_PI;
		offsetScore *= 1 / (1 + ((pos - trackedPosition.head<2>()) / 10.0f).squaredNorm() + rotationOffset*rotationOffset); // (10.0f) 1cm offset -> 0.5 score
		offsetScore *= std::max((float)blobAmount / 5.f, trackedScore);

		TrackedBotHypothesis::recalcPostColorCalib(r);
	}

	void recalcPostColorCalib(const Resources &r) override {
		score = offsetScore;
		calcTrackingScore(r);
	}

private:
	inline void calcTrackingScore(const Resources& r) {
		if(blobAmount < 2) {
			score = 0.0f;
			return;
		}

		float blobScore = 1.0f;
		for(int i = 0; i < 5; i++) {
			const Match* const& blob = blobs[i];
			if(blob == nullptr)
				continue;

			Eigen::Vector3i blobColor;
			Eigen::Vector3i oppositeColor;
			if(i == 0) {
				blobColor = botId >= 16 ? r.blue : r.yellow;
				oppositeColor = botId >= 16 ? r.yellow : r.blue;
			} else {
				blobColor = ((patterns[botId % 16] >> (4-i)) & 1) ? r.green : r.pink;
				oppositeColor = ((patterns[botId % 16] >> (4-i)) & 1) ? r.pink : r.green;
			}

			//blobScore *= 1 - (float)(blob->color - blobColor).norm() / 443.4050f; // sqrt(3 * 256**2)
			blobScore = std::min(blobScore, ((blob->color - oppositeColor).squaredNorm() - (blob->color - blobColor).squaredNorm() > 0 ? 1.0f : 0.1f));
		}

		score = std::max(0.f, score * blobScore);
	}

	const float trackedScore;
	const Eigen::Vector3f trackedPosition;
};

void generateCartesianTrackedBotHypotheses(const Resources& r, std::list<std::unique_ptr<BotHypothesis>>& bots, std::vector<Match>& matches, KDTree& blobs, const double currentTimestamp) {
	std::vector<Match*> botBlobs;
	for (const auto& camTracked : r.socket->getTrackedObjects()) { //TODO Concurrent Modification possible?
		for (const auto& tracked : camTracked.second) {
			if(tracked.id == -1)
				continue;

			auto timeDelta = (float)(currentTimestamp - tracked.timestamp);
			Eigen::Vector2f reprojectedPosition = r.perspective->model.image2field(r.perspective->model.field2image({tracked.x, tracked.y, tracked.z}), r.gcSocket->maxBotHeight).head<2>();
			Eigen::Vector3f trackedPosition = Eigen::Vector3f(reprojectedPosition.x(), reprojectedPosition.y(), tracked.w) + Eigen::Vector3f(tracked.vx, tracked.vy, tracked.vw) * timeDelta;

			timeDelta = std::min(timeDelta, 0.05f); //prevent runtime escalation when FPS drop below 20 FPS (likely due to excessive timeDelta)
			//Double acceleration due to velocity determination from two frame difference
			float blobSearchRadius = (float)r.maxBotAcceleration * timeDelta * timeDelta + r.perspective->field.max_robot_radius();

			float bestBotScore = 0.0f;
			std::unique_ptr<BotHypothesis> bestBot = nullptr;

			botBlobs.clear();
			botBlobs.push_back(nullptr);
			blobs.rangeSearch(botBlobs, trackedPosition.head<2>(), blobSearchRadius);
			for(Match* const& a : botBlobs) {
				for(Match* const& b : botBlobs) {
					if(b != nullptr && a == b)
						continue;

					for(Match* const& c : botBlobs) {
						if(c != nullptr && (a == c || b == c))
							continue;

						for(Match* const& d : botBlobs) {
							if(d != nullptr && (a == d || b == d || c == d))
								continue;

							for(Match* const& e : botBlobs) {
								if (e != nullptr && (a == e || b == e || c == e || d == e))
									continue;

								std::unique_ptr<BotHypothesis> bot = std::make_unique<TrackedBotHypothesis>(r, tracked, trackedPosition, a, b, c, d, e);
								if(bot->score > bestBotScore) {
									bestBotScore = bot->score;
									bestBot = std::move(bot);
								}
							}
						}
					}
				}
			}

			if(bestBot == nullptr)
				continue;

			bots.push_back(std::move(bestBot));
		}
	}
}

void generateCartesianBotHypotheses(const Resources& r, std::list<std::unique_ptr<BotHypothesis>>& bots, std::vector<Match>& matches, KDTree& blobs) {
	std::vector<Match*> botBlobs;
	for(int i = 0; i < blobs.getSize(); i++) {
		Match& blob = matches[i];

		float bestBotScore = 0.0f;
		std::unique_ptr<BotHypothesis> bestBot = nullptr;

		botBlobs.clear();
		blobs.rangeSearch(botBlobs, blob.pos, r.perspective->field.max_robot_radius());
		if(botBlobs.size() < 4)
			continue;

		for(Match* const& a : botBlobs) {
			for(Match* const& b : botBlobs) {
				if(a == b)
					continue;

				for(Match* const& c : botBlobs) {
					if(a == c || b == c)
						continue;

					for(Match* const& d : botBlobs) {
						if(a == d || b == d || c == d)
							continue;

						std::unique_ptr<BotHypothesis> bot = std::make_unique<DetectionBotHypothesis>(r, &blob, a, b, c, d);
						if(bot->score > bestBotScore) {
							bestBotScore = bot->score;
							bestBot = std::move(bot);
						}
					}
				}
			}
		}

		if(bestBot == nullptr)
			continue;

		bots.push_back(std::move(bestBot));
	}
}

void generateAngleSortedBotHypotheses(const Resources& r, std::list<std::unique_ptr<BotHypothesis>>& bots, std::vector<Match>& matches, KDTree& blobs) {
	std::vector<Match*> botBlobs;
	for(int i = 0; i < blobs.getSize(); i++) {
		Match& blob = matches[i];

		float bestBotScore = 0.0f;
		std::unique_ptr<BotHypothesis> bestBot = nullptr;

		botBlobs.clear();
		blobs.rangeSearch(botBlobs, blob.pos, r.perspective->field.max_robot_radius());
		if(botBlobs.size() < 4)
			continue;

		std::sort(botBlobs.begin(), botBlobs.end(), [&](const Match* a, const Match* b) -> bool {
			Eigen::Vector2f aDiff = a->pos - blob.pos;
			Eigen::Vector2f bDiff = b->pos - blob.pos;
			return atan2f(aDiff.y(), aDiff.x()) < atan2f(bDiff.y(), bDiff.x());
		});

		const int size = (int)botBlobs.size();
		for(int a = 0; a < size; a++) {
			for(int b = a+1; b < a+size-2; b++) {
				for(int c = b+1; c < a+size-1; c++) {
					for(int d = c+1; d < a+size; d++) {
						std::unique_ptr<BotHypothesis> bot = std::make_unique<DetectionBotHypothesis>(r, &blob, botBlobs[a], botBlobs[b%size], botBlobs[c%size], botBlobs[d%size]);
						if(bot->score > bestBotScore) {
							bestBotScore = bot->score;
							bestBot = std::move(bot);
						}
					}
				}
			}
		}

		bots.push_back(std::move(bestBot));
	}
}

void generateRadiusSearchTrackedBotHypotheses(const Resources& r, std::list<std::unique_ptr<BotHypothesis>>& bots, std::vector<Match>& matches, KDTree& blobs, const double currentTimestamp) {
	std::vector<Match*> botBlobs[5];
	for (const auto& camTracked : r.socket->getTrackedObjects()) { //TODO Concurrent Modification possible?
		for (const auto& tracked : camTracked.second) {
			if(tracked.id == -1)
				continue;

			auto timeDelta = (float)(currentTimestamp - tracked.timestamp);
			Eigen::Vector2f reprojectedPosition = r.perspective->model.image2field(r.perspective->model.field2image({tracked.x, tracked.y, tracked.z}), r.gcSocket->maxBotHeight).head<2>();
			Eigen::Vector3f trackedPosition = Eigen::Vector3f(reprojectedPosition.x(), reprojectedPosition.y(), tracked.w) + Eigen::Vector3f(tracked.vx, tracked.vy, tracked.vw) * timeDelta;
			Eigen::Rotation2Df rotation(trackedPosition.z());

			timeDelta = std::min(timeDelta, 0.05f); //prevent runtime escalation when FPS drop below 20 FPS (likely due to excessive timeDelta)
			//Double acceleration due to velocity determination from two frame difference
			float blobSearchRadius = (float)r.maxBotAcceleration * timeDelta * timeDelta + (float)r.minTrackingRadius;

			float bestBotScore = 0.0f;
			std::unique_ptr<BotHypothesis> bestBot = nullptr;

			for(int i = 0; i < 5; i++) {
				botBlobs[i].clear();
				botBlobs[i].push_back(nullptr);
				blobs.rangeSearch(botBlobs[i], trackedPosition.head<2>() + rotation * patternPos[i], blobSearchRadius);
			}

			for(Match* const& a : botBlobs[0]) {
				for(Match* const& b : botBlobs[1]) {
					if(b != nullptr && a == b)
						continue;

					for(Match* const& c : botBlobs[2]) {
						if(c != nullptr && (a == c || b == c))
							continue;

						for(Match* const& d : botBlobs[3]) {
							if(d != nullptr && (a == d || b == d || c == d))
								continue;

							for(Match* const& e : botBlobs[4]) {
								if (e != nullptr && (a == e || b == e || c == e || d == e))
									continue;

								std::unique_ptr<BotHypothesis> bot = std::make_unique<TrackedBotHypothesis>(r, tracked, trackedPosition, a, b, c, d, e);
								if(bot->score > bestBotScore) {
									bestBotScore = bot->score;
									bestBot = std::move(bot);
								}
							}
						}
					}
				}
			}

			if(bestBot == nullptr)
				continue;

			bots.push_back(std::move(bestBot));
		}
	}
}

template<typename T>
void filterHypothesesScore(std::list<std::unique_ptr<T>>& bots, float threshold) {
	for(auto it = bots.cbegin(); it != bots.cend(); ) {
		if((*it)->score <= threshold) {
			it = bots.erase(it);
		} else {
			it++;
		}
	}
}

void filterClippingBotBotHypotheses(std::list<std::unique_ptr<BotHypothesis>>& bots) {
	for (auto it1 = bots.cbegin(); it1 != bots.cend(); ) {
		const auto& bot1 = *it1;
		bool remove = false;
		for (auto it2 = bots.cbegin(); it2 != bots.cend(); it2++) {
			const auto& bot2 = *it1;
			if (bot2->score > bot1->score && bot1->isClipping(*bot2)) {
				remove = true;
				break;
			}
		}

		if(remove) {
			it1 = bots.erase(it1);
			continue;
		}

		for (auto it2 = bots.cbegin(); it2 != bots.cend(); ) {
			const auto& bot2 = *it2;
			if (bot2->score <= bot1->score && bot1->isClipping(*bot2) && it1 != it2) {
				it2 = bots.erase(it2);
			} else {
				it2++;
			}
		}

		it1++;
	}
}

void generateNonclippingBallHypotheses(const Resources& r, const std::list<std::unique_ptr<BotHypothesis>>& bots, std::vector<Match>& matches, std::list<std::unique_ptr<BallHypothesis>>& balls) {
	for (const auto& match : matches) {
		std::unique_ptr<BallHypothesis> ball = std::make_unique<BallHypothesis>(r, &match);
		bool nextToBot = false;
		for (const auto& bot : bots) {
			if (bot->isClipping(r, *ball)) {
				nextToBot = true;
				break;
			}
		}

		if(nextToBot)
			continue;

		balls.push_back(std::move(ball));
	}
}

/*static void bgrDrawBlobs(const Resources& r, Image& bgr, const std::list<CLMatch>& matches, const RGB& color) {
	auto bgrMap = bgr.cvReadWrite();

	for(const CLMatch& match : matches) {
		V2 pos = r.perspective->field2image({(float)match.x, (float)match.y, r.gcSocket->maxBotHeight});
		cv::drawMarker(*bgrMap, cv::Point(2*pos.x, 2*pos.y), CV_RGB(color.r, color.g, color.b), cv::MARKER_CROSS, 10);
		//cv::putText(*bgrMap, std::to_string((int)(match.score*100)) + " h" + std::to_string((int)hsv.r) + "s" + std::to_string((int)hsv.g) + "v" + std::to_string((int)hsv.b), cv::Point(2*pos.x, 2*pos.y), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(color.r, color.g, color.b));
	}
}*/

static inline void updateColor(const Resources& r, const Eigen::Vector3i& reference, const Eigen::Vector3i& oldColor, Eigen::Vector3i& color) {
	const float updateForce = 1.0f - r.referenceForce - r.historyForce;
	color = (r.referenceForce*reference.cast<float>() + r.historyForce*oldColor.cast<float>() + updateForce*color.cast<float>()).cast<int>();
}

static void updateColors(Resources& r, const std::list<std::unique_ptr<BotHypothesis>>& bestBotModels, const std::list<std::unique_ptr<BallHypothesis>>& ballCandidates) {
	Eigen::Vector3i oldFalseOrange = r.falseOrange;
	Eigen::Vector3i oldOrange = r.orange;
	Eigen::Vector3i oldYellow = r.yellow;
	Eigen::Vector3i oldBlue = r.blue;
	Eigen::Vector3i oldGreen = r.green;
	Eigen::Vector3i oldPink = r.pink;

	std::vector<Eigen::Vector3i> centerBlobs;
	Eigen::Vector3i pink(0, 0, 0);
	int pinkN = 0;
	Eigen::Vector3i green(0, 0, 0);
	int greenN = 0;
	for (const auto& model : bestBotModels) {
		if(model->blobs[0] != nullptr)
			centerBlobs.push_back(model->blobs[0]->color);

		int botId = model->botId % 16;
		for(int i = 1; i < 5; i++) {
			const Match* blob = model->blobs[i];
			if(blob == nullptr)
				continue;

			if((patterns[botId] >> (4-i)) & 1) {
				green += blob->color;
				greenN++;
			} else {
				pink += blob->color;
				pinkN++;
			}
		}
	}

	if(pinkN > 0) {
		r.pink = pink / pinkN;
		updateColor(r, r.pinkReference, oldPink, r.pink);
	}
	if(greenN > 0) {
		r.green = green / greenN;
		updateColor(r, r.greenReference, oldGreen, r.green);
	}

	if(kMeans(r.pink, centerBlobs, r.yellow, r.blue)) {
		updateColor(r, r.yellowReference, oldYellow, r.yellow);
		updateColor(r, r.blueReference, oldBlue, r.blue);
	}

	std::vector<Eigen::Vector3i> ballBlobs;
	for (const auto& ball : ballCandidates)
		ballBlobs.push_back(ball->blob->center);

	if(kMeans(r.blue, ballBlobs, r.orange, r.falseOrange)) {
		updateColor(r, r.orangeReference, oldOrange, r.orange);
		updateColor(r, r.falseOrangeReference, oldFalseOrange, r.falseOrange);
	}
}

static volatile bool noSigterm = true;
void sig_stop(int sig_num) {
	noSigterm = false;
}

int main(int argc, char* argv[]) {
	signal(SIGTERM, sig_stop);
	signal(SIGINT, sig_stop);
	Resources r(YAML::LoadFile(argc > 1 ? argv[1] : "config.yml"));

	cl::Kernel rggb2img = r.openCl->compile(kernel_rggb2img_cl, kernel_rggb2img_cl_end);
	cl::Kernel bgr2img = r.openCl->compile(kernel_bgr2img_cl, kernel_bgr2img_cl_end);
	cl::Kernel perspectiveKernel = r.openCl->compile(kernel_perspective_cl, kernel_perspective_cl_end);
	cl::Kernel colorKernel = r.openCl->compile(kernel_color_cl, kernel_color_cl_end);
	cl::Kernel satHorizontalKernel = r.openCl->compile(kernel_satHorizontal_cl, kernel_satHorizontal_cl_end);
	cl::Kernel satVerticalKernel = r.openCl->compile(kernel_satVertical_cl, kernel_satVertical_cl_end);
	cl::Kernel circleKernel = r.openCl->compile(kernel_satCircle_cl, kernel_satCircle_cl_end);
	cl::Kernel matchKernel = r.openCl->compile(kernel_matches_cl, kernel_matches_cl_end);

	while(r.waitForGeometry && !r.socket->getGeometryVersion()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		r.socket->geometryCheck();
	}

	uint32_t frameId = 0;
	while(noSigterm) {
		frameId++;
		std::shared_ptr<Image> img = r.camera->readImage();
		if(img == nullptr)
			break;

		double startTime = r.camera->getTime();
		double realStartTime = getRealTime(); // Just for realtime performance measurements
		r.socket->geometryCheck();
		r.perspective->geometryCheck(r.cameraAmount, img->width, img->height, r.gcSocket->maxBotHeight);

		std::shared_ptr<CLImage> clImg = r.openCl->acquire(&PixelFormat::RGBA8, img->width, img->height, img->name);
		//TODO better type switching
		OpenCL::await(img->format == &PixelFormat::RGGB8 ? rggb2img : bgr2img, cl::EnqueueArgs(cl::NDRange(clImg->width, clImg->height)), img->buffer, clImg->image);

		//std::shared_ptr<Image> mask = std::make_shared<Image>(&PixelFormat::U8, img->width, img->height);
		//OpenCL::wait(r.openCl->run(r.bgkernel, cl::EnqueueArgs(cl::NDRange(img->width, img->height)), img->buffer, bg->buffer, mask->buffer, img->format->stride, img->format->rowStride, (uint8_t)16)); //TODO adaptive threshold
		//bgsub->apply(*img->cvRead(), *mask->cvWrite());

		if(r.perspective->geometryVersion) {
			cl::NDRange visibleFieldRange(r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1]);
			std::shared_ptr<CLImage> flat = r.openCl->acquire(&PixelFormat::RGBA8, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(perspectiveKernel, cl::EnqueueArgs(visibleFieldRange), clImg->image, flat->image, r.perspective->getClPerspective(), (float)r.gcSocket->maxBotHeight, r.perspective->fieldScale, r.perspective->visibleFieldExtent[0], r.perspective->visibleFieldExtent[2]);
			//cv::GaussianBlur(flat.read<RGBA>().cv, blurred.write<RGBA>().cv, {5, 5}, 0, 0, cv::BORDER_REPLICATE);
			std::shared_ptr<CLImage> color = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(colorKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, color->image, (int)ceil(r.maxBlobRadius/r.perspective->fieldScale)/3);
			std::shared_ptr<CLImage> colorHor = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(satHorizontalKernel, cl::EnqueueArgs(cl::NDRange(r.perspective->reprojectedFieldSize[1])), color->image, colorHor->image);
			std::shared_ptr<CLImage> colorSat = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(satVerticalKernel, cl::EnqueueArgs(cl::NDRange(r.perspective->reprojectedFieldSize[0])), colorHor->image, colorSat->image);
			std::shared_ptr<CLImage> circ = r.openCl->acquire(&PixelFormat::F32, r.perspective->reprojectedFieldSize[0], r.perspective->reprojectedFieldSize[1], img->name);
			OpenCL::await(circleKernel, cl::EnqueueArgs(visibleFieldRange), colorSat->image, circ->image, (int)ceil(r.maxBlobRadius/r.perspective->fieldScale)); //TODO testing at robocup with MINBLOBRADIUS? COMPARE! , (int)floor(r.minBlobRadius/r.perspective->fieldScale)

			if(r.debugImages) {
				flat->save(".perspective." + std::to_string(frameId) + ".png");
				color->save(".color." + std::to_string(frameId) + ".png", 0.25f, 128.f);
				circ->save(".circle." + std::to_string(frameId) + ".png");
			}

			CLArray counter(sizeof(cl_int)*3);
			{
				CLMap<int> counterMap = counter.write<int>();
				counterMap[0] = 0;
				counterMap[1] = 0;
				counterMap[2] = 0;
			}
			CLArray matchArray(sizeof(CLMatch) * r.maxBlobs);
			OpenCL::await(matchKernel, cl::EnqueueArgs(visibleFieldRange), flat->image, circ->image, matchArray.buffer, counter.buffer, (float)r.minCircularity, (float)r.minScore, (int)floor(r.minBlobRadius/r.perspective->fieldScale), r.maxBlobs); //TODO borked minScore and radius at robocup?
			//std::cout << "[match filtering] time " << (getTime() - startTime) * 1000.0 << " ms" << std::endl;

			std::vector<Match> matches; //Same lifetime as KDTree required
			{
				CLMap<int> counterMap = counter.read<int>();
				CLMap<CLMatch> matchMap = matchArray.read<CLMatch>();
				//std::cerr << (flat->width*flat->height - (counterMap[2] + counterMap[1] + counterMap[0])) << "circScore " << counterMap[2] << "circPeak " << counterMap[1] << "score " << counterMap[0] << std::endl;
				const int matchAmount = std::min(r.maxBlobs, counterMap[0]);
				matches.reserve(matchAmount);

				for(int i = 0; i < matchAmount; i++) {
					CLMatch& match = matchMap[i];
					matches.push_back({
						.pos = r.perspective->flat2field({match.x, match.y}),
						.color = {match.color.r, match.color.g, match.color.b},
						.center = {match.center.r, match.center.g, match.center.b},
						.circ = match.circ,
						.score = match.score
					});
				}

				if(counterMap[0] > r.maxBlobs)
					std::cerr << "[blob] max blob amount reached: " << counterMap[0] << "/" << r.maxBlobs << std::endl;
			}

			KDTree blobs = matches.empty() ? KDTree() : KDTree(&matches[0]);
			for(int i = 1; i < matches.size(); i++)
				blobs.insert(&matches[i]);

			std::list<std::unique_ptr<BotHypothesis>> botHypotheses;
			//generateCartesianTrackedBotHypotheses(r, botHypotheses, matches, blobs, startTime);
			generateRadiusSearchTrackedBotHypotheses(r, botHypotheses, matches, blobs, startTime);
			//generateCartesianBotHypotheses(r, botHypotheses, matches, blobs);
			generateAngleSortedBotHypotheses(r, botHypotheses, matches, blobs);
			filterHypothesesScore(botHypotheses, r.minBotConfidence);
			filterClippingBotBotHypotheses(botHypotheses);
			std::list<std::unique_ptr<BallHypothesis>> ballHypotheses;
			generateNonclippingBallHypotheses(r, botHypotheses, matches, ballHypotheses);

			updateColors(r, botHypotheses, ballHypotheses);
			for (auto& bot : botHypotheses)
				bot->recalcPostColorCalib(r);
			for (auto& ball : ballHypotheses)
				ball->recalcPostColorCalib(r);

			filterHypothesesScore(ballHypotheses, 0.0);

			//TODO area around tracked ball with reduced or alternative minCircularity?
			//TODO test circ score scoring influences
			//TODO more than one ball on field line unrealistic

			SSL_WrapperPacket wrapper;
			SSL_DetectionFrame* detection = wrapper.mutable_detection();
			detection->set_frame_number(frameId);
			detection->set_t_capture(startTime);
			if(img->timestamp != 0)
				detection->set_t_capture_camera(img->timestamp);
			detection->set_camera_id(r.camId);

			for (const auto& bot : botHypotheses)
				bot->addToDetectionFrame(r, detection);
			for (const auto& ball : ballHypotheses)
				ball->addToDetectionFrame(r, detection);

			detection->set_t_sent(r.camera->getTime());
			r.socket->send(wrapper);
			std::cout << "[main] time " << (getRealTime() - realStartTime) * 1000.0 << " ms " << blobs.getSize() << " blobs " << detection->balls().size() << " balls " << (detection->robots_yellow_size() + detection->robots_blue_size()) << " bots" << std::endl;

			if(r.rawFeed) {
				r.rtpStreamer->sendFrame(clImg);
			} else {
				switch(((long)(startTime/20.0) % 3)) {
					case 0:
						r.rtpStreamer->sendFrame(flat);
						break;
					case 1:
						r.rtpStreamer->sendFrame(color);
						break;
					case 2:
						r.rtpStreamer->sendFrame(circ);
						break;
				}
			}
		} else if(r.socket->getGeometryVersion()) {
			geometryCalibration(r, *img);
		} else {
			r.rtpStreamer->sendFrame(clImg);
		}
	}

	std::cout << "Stopping vision_processor" << std::endl;
	return 0;
}
