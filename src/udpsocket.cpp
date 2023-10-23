#include "udpsocket.h"
#include "messages_robocup_ssl_wrapper.pb.h"

#include <iostream>
#include <cstring>
#include <google/protobuf/util/message_differencer.h>

UDPSocket::UDPSocket(const std::string &ip, uint16_t port)
{
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

	receiver = std::make_unique<std::thread>(&UDPSocket::receiverRun, this);
}

UDPSocket::~UDPSocket()
{
#ifdef _WIN32
	WSACleanup();
#endif
}

void UDPSocket::send(google::protobuf::Message& msg) {
	std::string str;
	msg.SerializeToString(&str);
	if(sendto(socket_, str.data(), str.length(), 0, &addr_, sizeof(addr_)) < 0)
	{
		std::cerr << "[UDPSocket] UDP Frame send failed: " << strerror(errno) << " " << strerrorname_np(errno) << std::endl;
	}
}

void UDPSocket::recv(google::protobuf::Message& msg) {
	char msgbuf[65535];

	int bytesRead = read(socket_, msgbuf, 65535);
	if (bytesRead < 0) {
		std::cerr << "[UDPSocket] UDP Frame recv failed: " << strerror(errno) << " " << strerrorname_np(errno) << std::endl;
		return;
	}

	msg.ParseFromArray(msgbuf, bytesRead);
}

void UDPSocket::receiverRun() {
	std::cout << "[UDPSocket] Awaiting geometry" << std::endl;
	while(true) {
		SSL_WrapperPacket wrapper;
		recv(wrapper);

		if(!wrapper.has_geometry())
			continue;

		if(!google::protobuf::util::MessageDifferencer::Equals(geometry, wrapper.geometry())) {
			std::cout << "[UDPSocket] New geometry received" << std::endl;
			geometry.CopyFrom(wrapper.geometry());
			geometryVersion++;
		}
	}
}
