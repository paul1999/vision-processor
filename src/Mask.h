#pragma once

#include <memory>
#include <utility>

#include "RLEVector.h"
#include "Perspective.h"

class Mask {
public:
	Mask(std::shared_ptr<Perspective> perspective, double maxBotHeight): perspective(std::move(perspective)), maxBotHeight(maxBotHeight) {}

	void geometryCheck();
	std::vector<Run>& getRuns() { return runs.getRuns(); }

private:
	double maxBotHeight;
	std::shared_ptr<Perspective> perspective;

	RLEVector runs;
	int geometryVersion = 0;
};
