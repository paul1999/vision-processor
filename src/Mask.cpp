#include "Mask.h"

void Mask::geometryCheck() {
	if(geometryVersion == perspective->getGeometryVersion())
		return;

	geometryVersion = perspective->getGeometryVersion();
	mask.clear();

	double halfLength = perspective->getFieldLength()/2.0 + perspective->getBoundaryWidth();
	double halfWidth = perspective->getFieldWidth()/2.0 + perspective->getBoundaryWidth();
	for(int y = 0; y < perspective->getHeight(); y++) {
		for(int x = 0; x < perspective->getWidth(); x++) {
			V2 groundPos = perspective->image2field({(double)x, (double)y}, maxBotHeight);
			if(groundPos.x < -halfLength || groundPos.x > halfLength || groundPos.y < -halfWidth || groundPos.y > halfWidth)
				continue;

			mask.add(x, y);
		}
	}
}

std::vector<int> Mask::scanArea() {
	std::vector<int> result;
	for(const Run& run : mask.getRuns()) {
		for(int x = run.x; x < run.x+run.length; x++) {
			result.push_back(x);
			result.push_back(run.y);
		}
	}
	return result;
}
