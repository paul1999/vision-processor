#include "spinnakersource.h"

#ifdef SPINNAKER

class SpinnakerImage : public Image {
public:
	// Image size halfed (RGB resolution)
	explicit SpinnakerImage(const Spinnaker::ImagePtr& pImage): Image(RGGB8, (int)pImage->GetWidth() / 2, (int)pImage->GetHeight() / 2, pImage->GetTimeStamp() / 1e9, (unsigned char*)pImage->GetData()), pImage(pImage) {}
	~SpinnakerImage() override { pImage->Release(); }

private:
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
	std::vector<void*> bufferPtrs;
	for(int i = 0; i < 3; i++) {
		std::shared_ptr<Image> buffer = BufferImage::create(RGGB8, width/2, height/2);
		bufferPtrs.push_back(buffer->getData());
		buffers.push_back(buffer);
	}

	pCam->SetBufferOwnership(Spinnaker::SPINNAKER_BUFFER_OWNERSHIP_USER);
	pCam->SetUserBuffers(bufferPtrs.data(), buffers.size(), width*height);

	/*if (IsWritable(pCam->GevSCPSPacketSize)) {
		pCam->GevSCPSPacketSize.SetValue(9000);
	}*/

	pCam->BeginAcquisition();
}

std::shared_ptr<Image> SpinnakerSource::readImage() {
	return std::make_shared<SpinnakerImage>(pCam->GetNextImage());
}

SpinnakerSource::~SpinnakerSource() {
	pCam->EndAcquisition();
}

#endif

