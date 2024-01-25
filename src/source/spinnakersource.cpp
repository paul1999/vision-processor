#ifdef SPINNAKER

#include "spinnakersource.h"


class SpinnakerImage : public Image {
public:
	// Image size halfed (RGB resolution)
	// TODO timestamp unusable due to bad resolution: pImage->GetTimeStamp() / 1e9
	SpinnakerImage(SpinnakerSource& source, const Spinnaker::ImagePtr& pImage): Image(*source.borrow(pImage)), source(source), pImage(pImage) {}

	~SpinnakerImage() override {
		pImage->Release();
		source.restore(*this);
	}

private:
	SpinnakerSource& source;
	const Spinnaker::ImagePtr pImage;
};

SpinnakerSource::SpinnakerSource(int id) {
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
	//pCam->PixelFormat.SetValue(Spinnaker::PixelFormat_BayerRG8);
	//pCam->BlackLevelAutoBalance.SetValue(Spinnaker::BlackLevelAutoBalance_Continuous);
	pCam->AutoExposureMeteringMode.SetValue(Spinnaker::AutoExposureMeteringMode_Average);
	//TODO auto black level
	//pCam->AutoExposureControlPriority.SetValue(Spinnaker::AutoExposureControlPriority_ExposureTime);
	pCam->ExposureAuto.SetValue(Spinnaker::ExposureAuto_Once);
	pCam->BalanceWhiteAuto.SetValue(Spinnaker::BalanceWhiteAuto_Once);
	pCam->GainAuto.SetValue(Spinnaker::GainAuto_Once);
	pCam->GammaEnable.SetValue(false);
	pCam->TLStream.StreamBufferHandlingMode.SetValue(Spinnaker::StreamBufferHandlingMode_NewestOnly);
	pCam->TLStream.StreamBufferCountManual.SetValue(pCam->TLStream.StreamBufferCountManual.GetMin());
	pCam->AcquisitionResultingFrameRate.GetValue(); //TODO

	int width = pCam->WidthMax.GetValue();
	int height = pCam->HeightMax.GetValue();
	for(int i = 0; i < 3; i++) {
		std::shared_ptr<Image> buffer = std::make_shared<Image>(&PixelFormat::RGGB8, width/2, height/2);
		buffers[buffer] = std::make_unique<CLMap<uint8_t>>(buffer->write<uint8_t>());
	}

	std::vector<void*> bufferPtrs;
	for (auto& item: buffers)
		bufferPtrs.push_back(**item.second);

	pCam->SetBufferOwnership(Spinnaker::SPINNAKER_BUFFER_OWNERSHIP_USER);
	pCam->SetUserBuffers(bufferPtrs.data(), buffers.size(), width*height);

	/*if (IsWritable(pCam->GevSCPSPacketSize)) {
		pCam->GevSCPSPacketSize.SetValue(9000);
	}*/

	pCam->BeginAcquisition();
}

std::shared_ptr<Image> SpinnakerSource::readImage() {
	return std::make_shared<SpinnakerImage>(*this, pCam->GetNextImage());
}

SpinnakerSource::~SpinnakerSource() {
	pCam->EndAcquisition();
}

std::shared_ptr<Image> SpinnakerSource::borrow(const Spinnaker::ImagePtr& pImage) {
	void* data = pImage->GetData();
	for (auto& item : buffers) {
		if(**item.second == data) {
			item.second = nullptr;
			return item.first;
		}
	}

	std::cerr << "[Spinnaker] Did not get image with given buffer, creating new buffer; expect OpenCL performance degradation" << std::endl;
	std::shared_ptr<Image> image = std::make_shared<Image>(&PixelFormat::RGGB8, (int)pImage->GetWidth() / 2, (int)pImage->GetHeight() / 2, (unsigned char*)pImage->GetData());
	buffers[image] = nullptr;
	return image;
}

void SpinnakerSource::restore(const Image& image) {
	for (auto& item : buffers) {
		if(item.first->buffer == image.buffer) {
			item.second = std::make_unique<CLMap<uint8_t>>(item.first->write<uint8_t>());
			return;
		}
	}
}

#endif
