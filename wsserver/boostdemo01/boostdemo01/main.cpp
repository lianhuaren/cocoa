//
// async_tcp_echo_server.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2018 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <boost/asio.hpp>
#include "WebSocket.h"
#include "wsserver.h"

#include <sys/socket.h>
#include <netinet/in.h>
typedef socklen_t Socklen;
typedef int SOCKET;

using boost::asio::ip::tcp;
//const int max_length = 1024;

//#define WS_BUFLEN (100 * 1024)
//
//typedef struct WsServer WsServer;
//
//struct WsHandler
//{
////    WsServer *server;
////    int socket;
//    void *context;
//    char resourceName[WS_BUFLEN];
//    char buf[WS_BUFLEN];
//};
//
//struct WsServer
//{
////    int sock;
////    char dirName[80];
////    int port;
//    //    pthread_t thread;
//    int cont;
//    void (*onOpen)(WsHandler* ws, char* msg);
//    void (*onClose)(WsHandler* ws, char* msg);
//    void (*onMessage)(WsHandler* ws, unsigned char* data, int len);
//    void (*onError)(WsHandler* ws, char* msg);
//    void* context;
////    WsHandler* clientWs;
//};

/* #############################################################
##   H A N D L E    C L I E N T
############################################################# */

static void onOpenDefault(WsHandler* ws, char* msg)
{
}

static void onCloseDefault(WsHandler* ws, char* msg)
{
}

static void onMessageDefault(WsHandler* ws, unsigned char* data, int len)
{
}

static void onErrorDefault(WsHandler* ws, char* msg)
{
}

WsServer* wsCreate(
    void (*onOpen)(WsHandler*, char*),
    void (*onClose)(WsHandler*, char*),
    void (*onMessage)(WsHandler*, unsigned char*, int),
    void (*onError)(WsHandler*, char*),
    void* context, char* dir, int port
)
{
#ifdef _WIN32
    WSADATA wsaData;
    //µ⁄“ª∏ˆ≤Œ ˝Œ™WinSock∞Ê±æ∫≈£¨µÕ◊÷Ω⁄Œ™÷˜∞Ê±æ∫≈£¨∏ﬂ◊÷Ω⁄Œ™–ﬁ’˝∞Ê±æ∫≈£¨µ⁄∂˛∏ˆ≤Œ ˝Œ™WSADATA¿‡–Õµƒ÷∏’Î ≥ı ºªØ≥…π¶∑µªÿ0
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("socket  error");
    }
#endif

    WsServer* obj = (WsServer*)malloc(sizeof(WsServer));
    if (!obj)
    {
        return NULL;
    }
    obj->onOpen = (onOpen) ? onOpen : onOpenDefault;
    obj->onClose = (onClose) ? onClose : onCloseDefault;
    obj->onMessage = (onMessage) ? onMessage : onMessageDefault;
    obj->onError = (onError) ? onError : onErrorDefault;
    obj->context = context;

    return obj;
}

void wsDelete(WsServer* obj)
{
    if (obj)
    {
//        sclose(obj->sock);
        free(obj);
    }
}

WsHandler* handlerCreate()
{
    WsHandler* obj = (WsHandler*)malloc(sizeof(WsHandler));
    if (!obj)
    {
        return NULL;
    }
    memset(obj, 0, sizeof(WsHandler));
    return obj;
}

void handlerDelete(WsHandler* obj)
{
    if (obj)
    {
        free(obj);
    }
}



#ifndef TRUE
#define TRUE  1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#define strtok_s strtok_r


static void trace(char* fmt, ...)
{
    fprintf(stdout, "WsServer: ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
    fprintf(stdout, "\n");
}


static void error(char* fmt, ...)
{
    fprintf(stderr, "WsServer err: ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

static char* trim(char* str)
{
    char* end;

    // Trim leading space
    while (isspace(*str)) str++;

    if (*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) end--;

    // Write new null terminator
    *(end + 1) = 0;

    return str;
}






static char* header =
"HTTP/1.1 101 Switching Protocols\r\n"
"Upgrade: websocket\r\n"
"Connection: Upgrade\r\n"
"Sec-WebSocket-Accept: %s\r\n"
"\r\n";

static void outf(tcp::socket &sock, char* fmt, ...)
{
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, 255, fmt, args);
    va_end(args);
//    swrite(fd, buf, strlen(buf));
    boost::asio::write(sock, boost::asio::buffer(buf, strlen(buf)));
}

static int  sread(tcp::socket &sock, void* buf, int len)
{
//    return recv(s,(char *)buf,len,NULL);


  boost::system::error_code error;
  size_t length = sock.read_some(boost::asio::buffer(buf,len), error);
    
    return length;
}

static int getInt(tcp::socket &sock, int size)
{
    unsigned char buf[16];
    long v = 0;
    if (sread(sock, buf, size) < size)
        return -1;
    unsigned char* b = buf;
    while (size--)
    {
        v = (v << 8) + *b++;
    }
    return v;
}



int
my_read(tcp::socket &sock, char* ptr)
{
    static int    read_cnt = 0;
    static char* read_ptr;
    static char    read_buf[WS_BUFLEN];

    if (read_cnt <= 0) {
    again:
//        if ((read_cnt = sread(fd, read_buf, sizeof(read_buf))) < 0) {
//            if (errno == EINTR)
//                goto again;
//            return(-1);
//        }
//        else if (read_cnt == 0)
//            return(0);
//        char data[max_length] = {0};

      boost::system::error_code error;
        read_cnt = sock.read_some(boost::asio::buffer(read_buf,sizeof(read_buf)), error);
        if (read_cnt < 0) {
            if (errno == EINTR)
                goto again;
            return(-1);
        }
        else if (read_cnt == 0)
            return(0);
        read_ptr = read_buf;
    }

    read_cnt--;
    *ptr = *read_ptr++;
    return(1);
}

int
readline(tcp::socket &sock, void* vptr, size_t maxlen)
{
    int        n, rc;
    char    c, * ptr;

    ptr = (char *)vptr;
    for (n = 1; n < maxlen; n++) {
        if ((rc = my_read(sock, &c)) == 1) {
            *ptr++ = c;
            if (c == '\n')
                break;    /* newline is stored, like fgets() */
        }
        else if (rc == 0) {
            if (n == 1)
                return(0);    /* EOF, no data read */
            else
                break;        /* EOF, some data was read */
        }
        else
            return(-1);        /* error, errno set by read() */
    }

    *ptr = 0;    /* null terminate like fgets() */
    return(n);
}

static void handleClientWebsocket(WsServer *srv, WsHandler* ws, tcp::socket &sock);

bool handleClient(WsServer *srv, tcp::socket &sock)
{
//        WsHandler* ws = (WsHandler*)ctx;
//        WsServer* srv = ws->server;
    WsHandler* ws = handlerCreate();
    if (!ws)
    {
        error("Could not create client handler");
        return -1;
    }
    ws->sock = &sock;
    
    char* buf = ws->buf;
//    int sock = ws->socket;

    int wsrequest = FALSE;
    readline(sock, buf, WS_BUFLEN);
    //FILE* in = fdopen(sock, "r");

    //fgets(buf, WS_BUFLEN, in);
    char* str = trim(buf);
    char* savptr;
    char* tok = strtok_s(str, " ", &savptr);
    if (strcmp(tok, "GET") != 0)
    {
        outf(sock, "405 Method '%s' not supported by this server", tok);
    }
    else
    {
        tok = strtok_s(NULL, " ", &savptr);
        strncpy(ws->resourceName, tok, WS_BUFLEN);
        char keybuf[40];
        keybuf[0] = 0;
        //while (fgets(buf, 255, in))
        while(readline(sock, buf, 255))
        {
            char* str = trim(buf);
            int len = strlen(str);
            if (len == 0)
                break;
            //trace("%s", str);
            char* name = strtok_s(str, ": ", &savptr);
            char* value = strtok_s(NULL, ": ", &savptr);
            //trace("name:'%s' value:'%s'", name, value);
            if (strcmp(name, "Sec-WebSocket-Key") == 0)
            {
                char encbuf[128];
                snprintf(encbuf, 128, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", value);
                sha1hash64((unsigned char*)encbuf, strlen(encbuf), keybuf);
                wsrequest = TRUE;
            }
        }
        trace("ready to process: %d", wsrequest);
        if (wsrequest)
        {
            outf(sock, header, keybuf);

            handleClientWebsocket(srv, ws, sock);

        }
        else
        {
//                serveFile(ws);
        }
    }


//        sclose(sock);
//
        handlerDelete(ws);

    return 0;
}

static void handleClientWebsocket(WsServer *srv, WsHandler* ws, tcp::socket &sock)
{
//    int sock = ws->socket;

//    srv->clientWs = ws;
//
    srv->onOpen(ws, "onOpen");

    int buflen = 1024 * 1024;

    unsigned char* recvBuf = (unsigned char*)malloc(buflen);
    if (!recvBuf)
    {
    }

    while (1)
    {
        unsigned char b;
        if (sread(sock, &b, 1) < 0)
            break;
        int fin = b & 0x80;
        //int rsv1   = b & 0x40;
        //int rsv2   = b & 0x20;
        //int rsv3   = b & 0x10;
        int opcode = b & 0x0f;
        if (sread(sock, &b, 1) < 0)
            break;
        int hasMask = b & 0x80;
        long paylen = b & 0x7f;
        if (paylen == 126)
            paylen = getInt(sock, 2);
        else if (paylen == 127)
            paylen = getInt(sock, 8);
        unsigned char mask[4];
        if (hasMask)
        {
            if (sread(sock, mask, 4) < 4)
                break;;
        }


        //trace("fin: %d opcode:%d hasMask:%d len:%ld mask:%d", fin, opcode, hasMask, paylen, mask);

        if (paylen > buflen)
        {
            error("Buffer too small for data");
            paylen = buflen;
        }

        if (sread(sock, recvBuf, paylen) < 0)
        {
            error("Read payload");
            break;
        }

        if (hasMask)
        {
            int i;
            for (i = 0; i < paylen; i++)
                recvBuf[i] ^= mask[i % 4];
        }

        recvBuf[paylen] = '\0';


        srv->onMessage(ws, recvBuf, paylen);

    }


    free(recvBuf);
    srv->onClose(ws, "onClose");
//    srv->clientWs = NULL;


}


WsServer *g_ssvr;

void session(tcp::socket sock)
{
  try
  {
      handleClient(g_ssvr,sock);
      
      return;
//    for (;;)
//    {
//        char data[max_length] = {0};
//
//      boost::system::error_code error;
//      size_t length = sock.read_some(boost::asio::buffer(data), error);
//
//        std::cout << data << std::endl;
//
//      if (error == boost::asio::error::eof)
//        break; // Connection closed cleanly by peer.
//      else if (error)
//        throw boost::system::system_error(error); // Some other error.
//
//      boost::asio::write(sock, boost::asio::buffer(data, length));
//    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception in thread: " << e.what() << "\n";
  }
}

void server(boost::asio::io_context& io_context, unsigned short port)
{
  tcp::acceptor a(io_context, tcp::endpoint(tcp::v4(), port));
  for (;;)
  {
    std::thread(session, a.accept()).detach();
  }
}

std::list<WebSocket *> connections;

static void onOpen(WsHandler *ws, char *msg)
{

    WebSocket *conn = new WebSocket();
    conn->ws = ws;

    ws->context = conn;

    connections.push_back(conn);
    
}

static void onClose(WsHandler *ws, char *msg)
{

    
    WebSocket *conn = (WebSocket *) ws->context;
    connections.remove(conn);
    delete conn;

    ws->context = NULL;
}

static void onMessage(WsHandler *ws, unsigned char *data, int len)
{

    printf("%s\n",(char *)data);

    WebSocket *conn = (WebSocket *) ws->context;
    string str((char *)data);
    conn->ProcessData(str);
}

static void onError(WsHandler *ws, char *msg)
{
//    WebSocket *conn = (WebSocket *) ws->context;

}

int main(int argc, char* argv[])
{

    
  try
  {
//    if (argc != 2)
//    {
//      std::cerr << "Usage: blocking_tcp_echo_server <port>\n";
//      return 1;
//    }

      g_ssvr = wsCreate(onOpen, onClose, onMessage, onError, NULL, NULL, 0);
      
    boost::asio::io_context io_context;

    server(io_context, std::atoi("8800"));
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
