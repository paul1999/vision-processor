/*
     Copyright 2025 Paul Bergmann

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
#pragma once
#ifdef DC1394

#include "cameradriver.h"
#include <memory>
#include <dc1394/camera.h>

class DC1394Driver : public CameraDriver {
public:
	explicit DC1394Driver(unsigned int id);
	~DC1394Driver() override;

	std::shared_ptr<RawImage> readImage() override;

	const PixelFormat format() override;

	double expectedFrametime() override;
private:
	struct dc1394_t_deleter {
		void operator()(dc1394_t* dc1394) {
			dc1394_free(dc1394);
		}
	};

	struct dc1394camera_t_deleter {
		void operator()(dc1394camera_t* camera) {
			dc1394_camera_free(camera);
		}
	};

	std::unique_ptr<dc1394_t, dc1394_t_deleter> dc1394;
	std::unique_ptr<dc1394camera_t, dc1394camera_t_deleter> camera;
};

#endif
