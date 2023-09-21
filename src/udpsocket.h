#pragma once


//Adapted from https://gist.github.com/hostilefork/f7cae3dc33e7416f2dd25a402857b6c6

#ifdef _WIN32
#include <Winsock2.h> // before Windows.h, else Winsock 1 conflict
#include <Ws2tcpip.h> // needed for ip_mreq definition for multicast
#include <Windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

#include <vector>
#include <google/protobuf/message.h>

class UDPSocket
{
public:
	UDPSocket(const std::string& ip, uint16_t port);
	~UDPSocket();

	void send(google::protobuf::Message& msg);
private:
	int socket_;
	struct sockaddr addr_;
};