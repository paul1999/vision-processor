#pragma once

#include <memory>
#include <utility>

#include "RLEVector.h"
#include "Perspective.h"

class Mask {
public:
	Mask(std::shared_ptr<Perspective> perspective, double maxBotHeight, double ballRadius): perspective(std::move(perspective)), maxBotHeight(maxBotHeight), ballRadius(ballRadius) {}

	void geometryCheck();
	std::shared_ptr<CLArray> scanArea(AlignedArrayPool& arrayPool);

	RLEVector& getRuns() { return mask; }

	double fieldExtentX[2];
	double fieldExtentY[2];

	float fieldScale = 5.0; //TODO autocalc
	int fieldSizeX = 1;
	int fieldSizeY = 1;
	std::shared_ptr<CLImage> flat = nullptr;
	std::shared_ptr<CLImage> color = nullptr;
	std::shared_ptr<CLImage> circ = nullptr;

private:
	double ballRadius;
	double maxBotHeight;
	std::shared_ptr<Perspective> perspective;

	RLEVector mask;
	int geometryVersion = 0;
};
