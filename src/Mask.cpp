#include "Mask.h"

static void addForHeight(RLEVector& mask, Perspective& perspective, const double height) {
	double halfLength = perspective.getFieldLength()/2.0 + perspective.getBoundaryWidth();
	double halfWidth = perspective.getFieldWidth()/2.0 + perspective.getBoundaryWidth();
	for(int y = 0; y < perspective.getHeight(); y++) {
		bool inRun = false;
		int runStart = 0;
		for(int x = 0; x < perspective.getWidth(); x++) {
			V2 groundPos = perspective.image2field({(double)x, (double)y}, height);
			if(groundPos.x < -halfLength || groundPos.x > halfLength || groundPos.y < -halfWidth || groundPos.y > halfWidth) {
				if (inRun) {
					mask.add({runStart, y, x - runStart});
					inRun = false;
				}
			} else {
				if(!inRun) {
					runStart = x;
					inRun = true;
				}
			}
		}

		if (inRun)
			mask.add({runStart, y, perspective.getWidth() - runStart});
	}
}

static void updateExtent(Mask& mask, const V2 point) {
	if(point.x < mask.fieldExtentX[0])
		mask.fieldExtentX[0] = point.x;
	if(point.x > mask.fieldExtentX[1])
		mask.fieldExtentX[1] = point.x;

	if(point.y < mask.fieldExtentY[0])
		mask.fieldExtentY[0] = point.y;
	if(point.y > mask.fieldExtentY[1])
		mask.fieldExtentY[1] = point.y;
}

void Mask::geometryCheck() {
	if(geometryVersion == perspective->getGeometryVersion())
		return;

	geometryVersion = perspective->getGeometryVersion();
	mask.clear();

	addForHeight(mask, *perspective, ballRadius);
	addForHeight(mask, *perspective, maxBotHeight);

	{
		V2 start = perspective->image2field({0.0, 0.0}, maxBotHeight);
		fieldExtentX[0] = start.x;
		fieldExtentX[1] = start.x;
		fieldExtentY[0] = start.y;
		fieldExtentY[1] = start.y;
	}

	for(int x = 0; x < perspective->getWidth(); x++)
		updateExtent(*this, perspective->image2field({(double)x, 0.0}, maxBotHeight));
	for(int x = 0; x < perspective->getWidth(); x++)
		updateExtent(*this, perspective->image2field({(double)x, perspective->getHeight() - 1.0}, maxBotHeight));

	for(int y = 0; y < perspective->getHeight(); y++)
		updateExtent(*this, perspective->image2field({0.0, (double)y}, maxBotHeight));
	for(int y = 0; y < perspective->getHeight(); y++)
		updateExtent(*this, perspective->image2field({perspective->getWidth() - 1.0, (double)y}, maxBotHeight));

	double halfLength = perspective->getFieldLength()/2.0 + perspective->getBoundaryWidth();
	double halfWidth = perspective->getFieldWidth()/2.0 + perspective->getBoundaryWidth();
	if(fieldExtentX[0] < -halfLength)
		fieldExtentX[0] = -halfLength;
	if(fieldExtentX[1] > halfLength)
		fieldExtentX[1] = halfLength;
	if(fieldExtentY[0] < -halfWidth)
		fieldExtentY[0] = -halfWidth;
	if(fieldExtentY[1] > halfWidth)
		fieldExtentY[1] = halfWidth;

	fieldSizeX = (fieldExtentX[1] - fieldExtentX[0]) / fieldScale;
	fieldSizeY = (fieldExtentY[1] - fieldExtentY[0]) / fieldScale;
	flat = std::make_shared<CLImage>(fieldSizeX, fieldSizeY, false);
	color = std::make_shared<CLImage>(fieldSizeX, fieldSizeY, true);
	circ = std::make_shared<CLImage>(fieldSizeX, fieldSizeY, true);
}

std::shared_ptr<CLArray> Mask::scanArea(AlignedArrayPool& arrayPool) {
	return mask.scanArea(arrayPool);
}
