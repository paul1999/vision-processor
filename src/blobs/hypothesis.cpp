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
#include "hypothesis.h"
#include "kmeans.h"
#include "pattern.h"


//https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
uint8_t Rgb2Hue(const Eigen::Vector3i& rgb) {
	unsigned char rgbMin, rgbMax;

	rgbMin = rgb.minCoeff();
	rgbMax = rgb.maxCoeff();

	uint8_t value = rgbMax;
	if (value == 0)
		return 0;

	uint8_t saturation = (uint8_t)(255 * long(rgbMax - rgbMin) / value);
	if (saturation == 0)
		return 0;

	if (rgbMax == rgb.x())
		return 0 + 43 * (rgb.y() - rgb.z()) / (rgbMax - rgbMin);
	else if (rgbMax == rgb.y())
		return 85 + 43 * (rgb.z() - rgb.x()) / (rgbMax - rgbMin);
	else
		return 171 + 43 * (rgb.x() - rgb.y()) / (rgbMax - rgbMin);
}


BallHypothesis::BallHypothesis(const Resources& r, const Match* blob): blob(blob), pos(blob->pos) {
	calcColorScore(r);
}

void BallHypothesis::recalcPostColorCalib(const Resources& r) {
	score = 1.0f;
	calcColorScore(r);
}

void BallHypothesis::addToDetectionFrame(const Resources& r, SSL_DetectionFrame* detection) {
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

void BallHypothesis::calcColorScore(const Resources& r) {
	//TODO test center vs color (updateColors as well)
	int falseOrange = (blob->center - r.field).squaredNorm();
	int orange = (blob->center - r.orange).squaredNorm();

	if (falseOrange <= orange) {
		score = 0;
		return;
	}

	score *= 1 - (float)orange / (float)falseOrange;
}


BotHypothesis::BotHypothesis(const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): blobs{a, b, c, d, e} {
	for(auto& blob : blobs)
		if(blob != nullptr)
			blobAmount++;

	calcPos();
	calcOffsetScore();
}

bool BotHypothesis::isClipping(const BotHypothesis& other) const {
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

bool BotHypothesis::isClipping(const Resources& r, const BallHypothesis& ball) const {
	const float clippedBallRadius = 0.48837f * r.perspective->field.ball_radius(); // a ball may clip up to 20% of the top-view area into the robot
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

void BotHypothesis::addToDetectionFrame(const Resources& r, SSL_DetectionFrame* detection) {
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

void BotHypothesis::calcPos() {
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

void BotHypothesis::calcOffsetScore() {
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


DetectionBotHypothesis::DetectionBotHypothesis(const Resources& r, const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): BotHypothesis(a, b, c, d, e) {
	calcBotId(r);
}

void DetectionBotHypothesis::recalcPostColorCalib(const Resources& r) {
	calcBotId(r);
}

void DetectionBotHypothesis::calcBotId(const Resources& r) {
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

HueBotHypothesis::HueBotHypothesis(const Resources& r, const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): BotHypothesis(a, b, c, d, e) {
	calcBotId(r);
}

void HueBotHypothesis::recalcPostColorCalib(const Resources& r) {}

void HueBotHypothesis::calcBotId(const Resources& r) {
	uint8_t hues[5] = {
			Rgb2Hue(blobs[0]->color),
			Rgb2Hue(blobs[1]->color),
			Rgb2Hue(blobs[2]->color),
			Rgb2Hue(blobs[3]->color),
			Rgb2Hue(blobs[4]->color)
	};
	botId = (abs((int8_t)(hues[0] - r.blueHue)) < abs((int8_t)(hues[0] - r.yellowHue)) ? 16 : 0) + patternLUT[
			(abs((int8_t)(hues[1] - r.greenHue)) < abs((int8_t)(hues[1] - r.pinkHue)) ? 8 : 0) +
			(abs((int8_t)(hues[2] - r.greenHue)) < abs((int8_t)(hues[2] - r.pinkHue)) ? 4 : 0) +
			(abs((int8_t)(hues[3] - r.greenHue)) < abs((int8_t)(hues[3] - r.pinkHue)) ? 2 : 0) +
			(abs((int8_t)(hues[4] - r.greenHue)) < abs((int8_t)(hues[4] - r.pinkHue)) ? 1 : 0)
	];
}


TrackedBotHypothesis::TrackedBotHypothesis(const Resources& r, const TrackingState& tracked, const Eigen::Vector3f& trackedPosition, const Match* a, const Match* b, const Match* c, const Match* d, const Match* e): BotHypothesis(a, b, c, d, e), trackedScore(tracked.confidence), trackedPosition(trackedPosition) {
	botId = tracked.id;

	float rotationOffset = remainderf(orientation - trackedPosition.z(), 2.0f * M_PI) / (float)M_PI;
	offsetScore *= 1 / (1 + ((pos - trackedPosition.head<2>()) / 10.0f).squaredNorm() + rotationOffset*rotationOffset); // (10.0f) 1cm offset -> 0.5 score
	offsetScore *= std::max((float)blobAmount / 5.f, trackedScore);

	TrackedBotHypothesis::recalcPostColorCalib(r);
}

void TrackedBotHypothesis::recalcPostColorCalib(const Resources& r) {
	score = offsetScore;
	calcTrackingScore(r);
}

void TrackedBotHypothesis::calcTrackingScore(const Resources& r) {
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