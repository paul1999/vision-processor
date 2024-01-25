#pragma once

#include <string>
#include <vector>

#include <google/protobuf/message.h>


class GroundTruth {
public:
	explicit GroundTruth(const std::string& source, int cameraId, double timestamp);

	const google::protobuf::Message& getMessage() const { return *message; }

private:
	std::unique_ptr<google::protobuf::Message> message;
};
