#include "udpsocket.h"
#include "proto/ssl_vision_wrapper.pb.h"
#include "proto/ssl_gc_referee_message.pb.h"

#include <cmath>
#include <iostream>
#include <cstring>
#include <google/protobuf/util/message_differencer.h>

UDPSocket::UDPSocket(const std::string& ip, uint16_t port) {
	//Adapted from https://gist.github.com/hostilefork/f7cae3dc33e7416f2dd25a402857b6c6

#ifdef _WIN32
	WSADATA wsaData;
    if (WSAStartup(0x0101, &wsaData))
    {
        std::cerr << "Failed to initialize Windows socket API" << std::endl;
        return;
    }
#endif

	socket_ = socket(AF_INET, SOCK_DGRAM, 0);
	if (socket_ < 0)
	{
		std::cerr << "Failed to open UDP socket" << std::endl;
		return;
	}

	auto* iNetAddr = (sockaddr_in*) &addr_;
	iNetAddr->sin_family = AF_INET;
	iNetAddr->sin_port = htons(port);
	if(!inet_aton(ip.c_str(), &iNetAddr->sin_addr))
	{
		std::cerr << "Invalid UDP target address " << ip << std::endl;
		return;
	}

	u_int yes = 1;
	if (setsockopt(socket_, SOL_SOCKET, SO_REUSEADDR, (char*) &yes, sizeof(yes)) < 0) {
		std::cerr << "Setting SO_REUSEADDR on UDP socket failed" << std::endl;
	}

	if(bind(socket_, &addr_, sizeof(addr_))) {
		std::cerr << "Could not bind to multicast socket" << std::endl;
	}

	struct ip_mreq mreq;
	inet_pton(AF_INET, ip.c_str(), &mreq.imr_multiaddr);
	inet_pton(AF_INET, "0.0.0.0", &mreq.imr_interface);
	if (setsockopt(socket_, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*) &mreq, sizeof(mreq)) < 0) {
		std::cerr << "Could not join multicast group" << std::endl;
	}

	receiver = std::thread(&UDPSocket::run, this);
}

UDPSocket::~UDPSocket() {
	closing = true;
	shutdown(socket_, SHUT_RD);
	::close(socket_);
	receiver.join();

#ifdef _WIN32
	WSACleanup();
#endif
}

void UDPSocket::send(const google::protobuf::Message& msg) {
	std::string str;
	msg.SerializeToString(&str);
	if(sendto(socket_, str.data(), str.length(), 0, &addr_, sizeof(addr_)) < 0) {
		std::cerr << "[UDPSocket] UDP Frame send failed: " << strerror(errno) << " " << strerrorname_np(errno) << std::endl;
	}
}

void UDPSocket::run() {
	while(true) {
		char msgbuf[65535];

		int bytesRead = read(socket_, msgbuf, 65535);
		if(closing)
			return;

		if (bytesRead < 0) {
			std::cerr << "[UDPSocket] UDP Frame recv failed: " << strerror(errno) << " " << strerrorname_np(errno) << std::endl;
			return;
		}

		parse(msgbuf, bytesRead);
	}
}


void VisionSocket::parse(char *data, int length) {
	SSL_WrapperPacket wrapper;
	wrapper.ParseFromArray(data, length);

	if(wrapper.has_detection()) {
		detectionTracking(wrapper.detection());
	}

	if(wrapper.has_geometry()) {
		if(!google::protobuf::util::MessageDifferencer::Equals(geometry, wrapper.geometry())) {
			std::cout << "[VisionSocket] New geometry received" << std::endl;
			geometry.CopyFrom(wrapper.geometry());
			geometryVersion++;
		}
	}
}

static inline void trackBots(const double timestamp, const float defaultBotHeight, const google::protobuf::RepeatedPtrField<SSL_DetectionRobot>& bots, const std::vector<TrackingState>& previous, std::vector<TrackingState>& objects, int idOffset) {
	for (const SSL_DetectionRobot& bot : bots) {
		float height = bot.has_height() ? bot.height() : defaultBotHeight;

		double nextDistance = INFINITY;
		const TrackingState* nextOldBot = nullptr;
		for(const TrackingState& oldBot : previous) {
			if(oldBot.id != bot.robot_id() + idOffset)
				continue;

			float dX = bot.x() - oldBot.x;
			float dY = bot.y() - oldBot.y;
			double distance = dX*dX + dY*dY;
			if(distance > nextDistance)
				continue;

			nextDistance = distance;
			nextOldBot = &oldBot;
		}

		if(nextOldBot == nullptr) {
			objects.push_back({
				(int)bot.robot_id() + idOffset, timestamp,
				bot.x(), bot.y(), height, bot.orientation(),
				0.0f, 0.0f, 0.0f, 0.0f
			});
		} else {
			float timeDelta = timestamp - nextOldBot->timestamp;
			objects.push_back({
				nextOldBot->id, timestamp,
				bot.x(), bot.y(), height, bot.orientation(),
				(bot.x() - nextOldBot->x) / timeDelta, (bot.y() - nextOldBot->y) / timeDelta, 0.0f, (bot.orientation() - nextOldBot->w) / timeDelta
			});
		}
	}
}

void VisionSocket::detectionTracking(const SSL_DetectionFrame &detection) {
	const double timestamp = detection.t_capture();

	const std::vector<TrackingState>& previous = trackedObjects[detection.camera_id()];
	std::vector<TrackingState> objects;

	//TODO better filtering than next
	for (const auto& ball : detection.balls()) {
		float z = ball.has_z() ? ball.z() : ballRadius;

		double nextDistance = INFINITY;
		const TrackingState* nextOldBall = nullptr;
		for(const TrackingState& oldBall : previous) {
			if(oldBall.id != -1)
				continue;

			float dX = ball.x() - oldBall.x;
			float dY = ball.y() - oldBall.y;
			float dZ = z - oldBall.z;
			double distance = dX*dX + dY*dY + dZ*dZ;
			if(distance > nextDistance)
				continue;

			nextDistance = distance;
			nextOldBall = &oldBall;
		}

		if(nextOldBall == nullptr) {
			objects.push_back({
				-1, timestamp,
				ball.x(), ball.y(), z, 0.0f,
				0.0f, 0.0f, 0.0f, 0.0f
			});
		} else {
			float timeDelta = timestamp - nextOldBall->timestamp;
			objects.push_back({
				-1, timestamp,
				ball.x(), ball.y(), z, 0.0f,
				(ball.x() - nextOldBall->x) / timeDelta, (ball.y() - nextOldBall->y) / timeDelta, (z - nextOldBall->z) / timeDelta, 0.0f
			});
		}
	}

	trackBots(timestamp, defaultBotHeight, detection.robots_yellow(), previous, objects, 0);
	trackBots(timestamp, defaultBotHeight, detection.robots_blue(), previous, objects, 16);

	trackedObjects[detection.camera_id()] = objects; //TODO synchronization/mutex
}


GCSocket::GCSocket(const std::string &ip, uint16_t port, const std::map<std::string, double>& botHeights): UDPSocket(ip, port), botHeights(botHeights), defaultBotHeight(0), maxBotHeight(0) {
	for (const auto& entry : botHeights) {
		if(entry.second > maxBotHeight)
			maxBotHeight = entry.second;

		defaultBotHeight += entry.second;
	}
	defaultBotHeight /= botHeights.size();
	yellowBotHeight = defaultBotHeight;
	blueBotHeight = defaultBotHeight;
}

void GCSocket::parse(char *data, int length) {
	Referee referee;
	referee.ParseFromArray(data, length);

	if(botHeights.find(referee.yellow().name()) != botHeights.end() && botHeights[referee.yellow().name()] != yellowBotHeight) {
		yellowBotHeight = botHeights[referee.yellow().name()];
		std::cout << "[GCSocket] Updated yellow bot height to " << yellowBotHeight << "mm" << std::endl;
	}

	if(botHeights.find(referee.blue().name()) != botHeights.end() && botHeights[referee.blue().name()] != blueBotHeight) {
		blueBotHeight = botHeights[referee.blue().name()];
		std::cout << "[GCSocket] Updated blue bot height to " << blueBotHeight << "mm" << std::endl;
	}
}