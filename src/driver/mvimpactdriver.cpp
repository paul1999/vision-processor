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
#ifdef MVIMPACT

#include "mvimpactdriver.h"
#include <mvIMPACT_CPP/mvIMPACT_acquire_GenICam.h>


class MVImpactImage : public Image {
public:
	explicit MVImpactImage(const std::shared_ptr<Request>& request): Image(&PixelFormat::RGGB8, request->imageWidth.read(), request->imageHeight.read(), (double)request->infoTimeStamp_us.read() / 1e6, (uint8_t*)request->imageData.read()), request(request) {}

private:
	std::shared_ptr<Request> request;
};


MVImpactDriver::MVImpactDriver(const int id): devMgr(mvIMPACT::acquire::DeviceManager()) {
	while(devMgr.deviceCount() <= id) {
		std::cerr << "[mvIMPACT] Waiting for cam: " << devMgr.deviceCount() << "/" << (id+1) << std::endl;
		devMgr.updateDeviceList();
		sleep(1);
	}

	device = devMgr[id];

	try {
		device->open();
	} catch(mvIMPACT::acquire::ImpactAcquireException& e) {
		std::cerr << "[mvIMPACT] Error while opening the camera: " << e.getErrorCodeAsString() << " " << e.getErrorString() << std::endl;
		exit(1);
	}

	GenICam::ImageFormatControl control(device);
	control.pixelFormat.writeS("BayerRG8");
	control.mvSensorDigitizationBitDepth.writeS("Bpp10");

	ImageProcessing proc(device);
	proc.restoreDefault();

	ImageDestination idest(device);
	idest.restoreDefault();

	//TODO user supplied memory functionality to prevent copies https://assets.balluff.com/documents/DRF_957352_AA_000/CaptureToUserMemory_8cpp-example.html

	provider = std::make_unique<mvIMPACT::acquire::helper::RequestProvider>(device);
	provider->acquisitionStart();
}

MVImpactDriver::~MVImpactDriver() {
	//delete fi;
	provider->acquisitionStop();
	device->close();
}

std::shared_ptr<Image> MVImpactDriver::readImage() {
	std::shared_ptr<Request> request = provider->waitForNextRequest();

	//Get only newest frame
	std::shared_ptr<Request> newerRequest;
	while((newerRequest = provider->waitForNextRequest(0)) != nullptr)
		request = newerRequest;

	if(!request->isOK()) {
		std::cerr << "[mvIMPACT] Error while acquiring image: " << request->requestResult.readS() << std::endl;
		return nullptr;
	}

	return std::make_shared<MVImpactImage>(request);
}

#endif

