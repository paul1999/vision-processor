#include "udpsocket.h"

#include <iostream>
#include <cstring>

UDPSocket::UDPSocket(const std::string &ip, uint16_t port)
{
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

	struct ip_mreq mreq;
	mreq.imr_multiaddr.s_addr = inet_addr(ip.c_str());
	mreq.imr_interface.s_addr = htonl(INADDR_ANY);
	if (setsockopt(socket_, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*) &mreq, sizeof(mreq)) < 0) {
		std::cerr << "Could not join multicast group" << std::endl;
	}
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
		std::cerr << "UDP Frame send failed: " << strerror(errno) << " " << strerrorname_np(errno) << std::endl;
	}
}

void UDPSocket::recv(google::protobuf::Message& msg) {
	char msgbuf[65535];
	unsigned int addrlen = sizeof(addr_);

	long bytesRead = recvfrom(socket_, msgbuf, 65535, 0, &addr_, &addrlen);
	if (bytesRead < 0) {
		std::cerr << "UDP Frame recv failed: " << strerror(errno) << " " << strerrorname_np(errno) << std::endl;
		return;
	}

	msgbuf[bytesRead] = 0;
	msg.ParseFromString(msgbuf);
}
