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
#ifdef SPINNAKER

#include "spinnakerdriver.h"


class SpinnakerImage : public Image {
public:
	// TODO investigate unusable timestamp due to bad time resolution: pImage->GetTimeStamp() / 1e9
	SpinnakerImage(SpinnakerDriver& source, const Spinnaker::ImagePtr& pImage): Image(*source.borrow(pImage)), source(source), pImage(pImage) {}

	~SpinnakerImage() override {
		pImage->Release();
		source.restore(*this);
	}

private:
	SpinnakerDriver& source;
	const Spinnaker::ImagePtr pImage;
};

SpinnakerDriver::SpinnakerDriver(int id, double exposure, double gain, WhiteBalanceType wbType, const std::vector<double>& wbValues) {
	pSystem = Spinnaker::System::GetInstance();

	while(true) {
		Spinnaker::CameraList camList = pSystem->GetCameras();
		if (camList.GetSize() > id) {
			pCam = camList.GetByIndex(id);
			pCam->Init();
			std::cout << "[Spinnaker] Opened " << pCam->DeviceModelName.GetValue() << " - " << pCam->DeviceSerialNumber.GetValue().c_str() << std::endl;
			camList.Clear();
			break;
		}

		std::cerr << "[Spinnaker] Waiting for cam: " << camList.GetSize() << "/" << (id+1) << std::endl;

		camList.Clear();
		sleep(1);
	}

	pCam->TriggerMode.SetValue(Spinnaker::TriggerMode_Off);
	pCam->AcquisitionMode.SetValue(Spinnaker::AcquisitionMode_Continuous);
	pCam->PixelFormat.SetValue(Spinnaker::PixelFormat_BayerRG8);
	pCam->GammaEnable.SetValue(false);
	pCam->AcquisitionFrameRateEnable.SetValue(false);
	/*pCam->GammaEnable.SetValue(true);
	pCam->Gamma.SetValue(0.45);*/

	if(exposure == 0.0) {
		pCam->AutoExposureControlPriority.SetValue(Spinnaker::AutoExposureControlPriority_Gain);
		pCam->AutoExposureMeteringMode.SetValue(Spinnaker::AutoExposureMeteringMode_Average);
		pCam->ExposureAuto.SetValue(Spinnaker::ExposureAuto_Continuous);
	} else {
		pCam->ExposureAuto.SetValue(Spinnaker::ExposureAuto_Off);
		pCam->ExposureTime.SetValue(exposure / 1000.0);
	}

	//TODO smarter autogain and autoexposure through feedback from blob brightness
	if(gain == 0.0) {
		pCam->GainAuto.SetValue(Spinnaker::GainAuto_Continuous);
	} else {
		pCam->GainAuto.SetValue(Spinnaker::GainAuto_Off);
		pCam->Gain.SetValue(gain);
	}

	if(wbType != WhiteBalanceType_Manual) {
		pCam->BalanceWhiteAutoProfile.SetValue(
				wbType == WhiteBalanceType_AutoOutdoor
				? Spinnaker::BalanceWhiteAutoProfile_Outdoor
				: Spinnaker::BalanceWhiteAutoProfile_Indoor
		);
		pCam->BalanceWhiteAuto.SetValue(Spinnaker::BalanceWhiteAuto_Continuous);
	} else {
		pCam->BalanceWhiteAuto.SetValue(Spinnaker::BalanceWhiteAuto_Off);
		pCam->BalanceRatioSelector.SetValue(Spinnaker::BalanceRatioSelector_Blue);
		pCam->BalanceRatio.SetValue(wbValues[0]);
		pCam->BalanceRatioSelector.SetValue(Spinnaker::BalanceRatioSelector_Red);
		pCam->BalanceRatio.SetValue(wbValues[1]);
	}

	pCam->TLStream.StreamBufferHandlingMode.SetValue(Spinnaker::StreamBufferHandlingMode_NewestOnly);
	pCam->TLStream.StreamBufferCountManual.SetValue(pCam->TLStream.StreamBufferCountManual.GetMin());

	// Provide image buffers to achieve faster mapping with OpenCL
	int width = pCam->WidthMax.GetValue();
	int height = pCam->HeightMax.GetValue();
	for(int i = 0; i < pCam->TLStream.StreamBufferCountManual.GetMin(); i++) {
		std::shared_ptr<Image> buffer = std::make_shared<Image>(&PixelFormat::RGGB8, width/2, height/2, "spinnaker");
		buffers[buffer] = std::make_unique<CLMap<uint8_t>>(buffer->write<uint8_t>());
	}

	std::vector<void*> bufferPtrs;
	for (auto& item: buffers)
		bufferPtrs.push_back(**item.second);

	pCam->SetBufferOwnership(Spinnaker::SPINNAKER_BUFFER_OWNERSHIP_USER);
	pCam->SetUserBuffers(bufferPtrs.data(), buffers.size(), width*height);

	/* TODO Advisable with Ethernet cameras on special interfaces
	 if (IsWritable(pCam->GevSCPSPacketSize)) {
		pCam->GevSCPSPacketSize.SetValue(9000);
	}*/

	pCam->BeginAcquisition();
}

std::shared_ptr<Image> SpinnakerDriver::readImage() {
	return std::make_shared<SpinnakerImage>(*this, pCam->GetNextImage());
}

double SpinnakerDriver::expectedFrametime() {
	return 1 / pCam->AcquisitionResultingFrameRate.GetValue();
}

SpinnakerDriver::~SpinnakerDriver() {
	pCam->EndAcquisition();
}

std::shared_ptr<Image> SpinnakerDriver::borrow(const Spinnaker::ImagePtr& pImage) {
	void* data = pImage->GetData();
	for (auto& item : buffers) {
		if(item.second != nullptr && **item.second == data) {
			item.second = nullptr;
			return item.first;
		}
	}

	std::cerr << "[Spinnaker] Did not get image with given buffer, creating new buffer; expect OpenCL performance degradation" << std::endl;
	std::shared_ptr<Image> image = std::make_shared<Image>(&PixelFormat::RGGB8, (int)pImage->GetWidth() / 2, (int)pImage->GetHeight() / 2, (unsigned char*)pImage->GetData());
	buffers[image] = nullptr;
	return image;
}

void SpinnakerDriver::restore(const Image& image) {
	for (auto& item : buffers) {
		if(item.first->buffer == image.buffer) {
			item.second = std::make_unique<CLMap<uint8_t>>(item.first->write<uint8_t>());
			return;
		}
	}
}

#endif
