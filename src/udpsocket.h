#pragma once


#include <thread>
#include <google/protobuf/message.h>
#include "messages_robocup_ssl_geometry.pb.h"
#include "messages_robocup_ssl_detection.pb.h"

#ifdef _WIN32
#include <Winsock2.h> // before Windows.h, else Winsock 1 conflict
#include <Ws2tcpip.h> // needed for ip_mreq definition for multicast
#include <Windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif


struct TrackingState {
	int id; // -1: ball, 0-15: yellow bot, 16-31: blue bot
	double timestamp;
	float x, y, z, w;
	float vx, vy, vz, vw;
};


class UDPSocket
{
public:
	UDPSocket(const std::string& ip, uint16_t port, float defaultBotHeight, float ballRadius);
	~UDPSocket();

	void send(google::protobuf::Message& msg);
	void close();

	int getGeometryVersion() const { return geometryVersion; }
	SSL_GeometryData& getGeometry() { return geometry; }

	std::map<int, std::vector<TrackingState>>& getTrackedObjects() { return trackedObjects; }
private:
	void receiverRun();
	void recv(google::protobuf::Message& msg) const;

	void detectionTracking(const SSL_DetectionFrame& detection);

	int socket_;
	struct sockaddr addr_;

	std::thread receiver;
	bool closing = false;

	const float defaultBotHeight;
	const float ballRadius;

	int geometryVersion = 0;
	SSL_GeometryData geometry;

	std::map<int, std::vector<TrackingState>> trackedObjects;
};