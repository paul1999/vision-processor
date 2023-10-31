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
	float x, y, z; // z is the orientation on bots
	float vx, vy, vz;
};


class UDPSocket
{
public:
	UDPSocket(const std::string& ip, uint16_t port); //TODO supply ball size
	~UDPSocket();

	void send(google::protobuf::Message& msg);

	int getGeometryVersion() const { return geometryVersion; }
	SSL_GeometryData& getGeometry() { return geometry; }

	void detectionTracking(const SSL_DetectionFrame& detection);
private:
	void receiverRun();
	void recv(google::protobuf::Message& msg) const;

	int socket_;
	struct sockaddr addr_;

	std::unique_ptr<std::thread> receiver;

	int geometryVersion = 0;
	SSL_GeometryData geometry;

	std::map<int, std::vector<TrackingState>> trackedObjects;
};