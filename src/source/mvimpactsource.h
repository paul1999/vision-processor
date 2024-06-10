#pragma once
#ifdef MVIMPACT

#include "videosource.h"
#include <mvIMPACT_CPP/mvIMPACT_acquire.h>
#include <apps/Common/exampleHelper.h> //Necessary header for compilation of <mvIMPACT_acquire_helper.h> without -fpermissive
#include <mvIMPACT_CPP/mvIMPACT_acquire_helper.h>

class MVImpactSource : public VideoSource {
public:
	explicit MVImpactSource(int id);
	~MVImpactSource() override;

	std::shared_ptr<Image> readImage() override;

private:
	mvIMPACT::acquire::DeviceManager devMgr;
	mvIMPACT::acquire::Device* device;
	std::unique_ptr<mvIMPACT::acquire::helper::RequestProvider> provider;
};

#endif
