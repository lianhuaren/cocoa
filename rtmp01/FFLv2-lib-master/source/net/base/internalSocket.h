#ifndef _FFL_SOCKET_DEF_H_
#define _FFL_SOCKET_DEF_H_

#include <FFL_Core.h>

#ifdef WIN32
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
typedef int  socklen_t;
#pragma comment(lib,"ws2_32.lib")

#define SOCKET_AGAIN WSAETIMEDOUT
#define SOCKET_ETIME WSAETIMEDOUT

#define SOCKET_ECONNRESET WSAECONNRESET
#define SOCKET_ERRNO() WSAGetLastError()
#define SOCKET_RESET(x) x=INVALID_SOCKET

#define FFL_SOCKET_CLOSE(x) \
       do{if(x!=INVALID_SOCKET){\
          closesocket(x); \
           x=INVALID_SOCKET;\
       }}while(0)

#define SOCKET_VALID(x) (x!=INVALID_SOCKET)
#define SOCKET_BUFF(x) ((char*)x)
//inline void internal_socket_setup(){
//    WORD version;
//	WSADATA wsaData;
//	version = MAKEWORD(1,1);
//	WSAStartup(version, &wsaData);
//}
//#define SOCKET_SETUP() internal_socket_setup()

#define SOCKET_SETUP() \
	do{WORD version;\
	WSADATA wsaData;\
	version = MAKEWORD(1,1);\
	WSAStartup(version, &wsaData);}while(0)

#define SOCKET_CLEANUP() WSACleanup()

#else
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/uio.h>

/*  需要重试一下 */
#define SOCKET_AGAIN EWOULDBLOCK
#define SOCKET_ETIME ETIMEDOUT
#define SOCKET_ECONNRESET ECONNRESET

#define SOCKET_ERRNO() errno
#define SOCKET_RESET(fd) fd = -1; (void)0
//#define FFL_SOCKET_CLOSE(fd)  if(fd>0) { close(fd);fd=0;} (void)0;
#define FFL_SOCKET_CLOSE(fd)  if(fd>0) { shutdown(fd,SHUT_RDWR);fd=0;} (void)0;
#define SOCKET_VALID(x) (x > 0)
#define SOCKET_SETUP() (void)0
#define SOCKET_CLEANUP() (void)0

#endif

#include <sys/types.h>
#include <errno.h>

#ifdef MACOSX
#include <poll.h>
#endif

#endif
