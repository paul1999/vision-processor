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
#pragma once


#include <thread>
#include <google/protobuf/message.h>
#include "proto/ssl_vision_geometry.pb.h"
#include "proto/ssl_vision_detection.pb.h"

#ifdef _WIN32
#include <Winsock2.h> // before Windows.h, else Winsock 1 conflict
#include <Ws2tcpip.h> // needed for ip_mreq definition for multicast
#include <Windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif


class UDPSocket {
public:
	UDPSocket(const std::string& ip, uint16_t port);
	~UDPSocket();

	void send(const google::protobuf::Message& msg);

private:
	virtual void parse(char* data, int length) = 0;
	void run();

	bool closing = false;
	int socket_;
	struct sockaddr addr_;

	std::thread receiver;
};


struct TrackingState {
	int id; // -1: ball, 0-15: yellow bot, 16-31: blue bot
	double timestamp;
	float x, y, z, w;
	float vx, vy, vz, vw;
};


class VisionSocket: public UDPSocket {
public:
	VisionSocket(const std::string &ip, uint16_t port, float defaultBotHeight, float ballRadius): UDPSocket(ip, port), defaultBotHeight(defaultBotHeight), ballRadius(ballRadius) {}

	void geometryCheck();
	int getGeometryVersion() const { return geometryVersion; }
	SSL_GeometryData& getGeometry() { return geometry; }

	std::map<int, std::vector<TrackingState>>& getTrackedObjects() { return trackedObjects; }
private:
	void parse(char* data, int length) override;

	void detectionTracking(const SSL_DetectionFrame& detection);

	const float defaultBotHeight;
	const float ballRadius;

	int geometryVersion = 0;
	SSL_GeometryData geometry;
	SSL_GeometryData receivedGeometry;
	std::mutex geometryMutex;

	std::map<int, std::vector<TrackingState>> trackedObjects;
};

class GCSocket: public UDPSocket {
public:
	GCSocket(const std::string &ip, uint16_t port, const std::map<std::string, double>& botHeights);

	double maxBotHeight;
	double defaultBotHeight;
	double yellowBotHeight;
	double blueBotHeight;

private:
	void parse(char* data, int length) override;

	std::map<std::string, double> botHeights;
};