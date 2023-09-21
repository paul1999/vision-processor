#include "spinnakersource.h"

#ifdef SPINNAKER

SpinnakerSource::SpinnakerSource(int id) {
	pSystem = Spinnaker::System::GetInstance();

	while(true) {
		Spinnaker::CameraList camList = pSystem->GetCameras();
		if (camList.GetSize() > id) {
			pCam = camList.GetByIndex(id);
			pCam->Init();
			fprintf(stderr, "Spinnaker: Opened %s - %s\n", pCam->DeviceModelName.GetValue().c_str(), pCam->DeviceSerialNumber.GetValue().c_str());
			camList.Clear();
			break;
		}

		fprintf(stderr, "Spinnaker: Number of cams: %u\n", camList.GetSize());

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
	pCam->AcquisitionResultingFrameRate.GetValue(); //TODO

	//width = pCam->WidthMax.GetValue();
	//height = pCam->HeightMax.GetValue();
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

