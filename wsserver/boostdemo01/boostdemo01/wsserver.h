/**
 * This is a very simple HTTP server that can also handle Websockets
 *
 * Authors:
 *   Bob Jamison
 *
 * Copyright (C) 2013 Bob Jamison
 * 
 *  This file is part of the SdrLib library.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 3 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */
#ifndef _WSSERVER_H_
#define _WSSERVER_H_
#include <boost/asio.hpp>
using boost::asio::ip::tcp;

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define WS_BUFLEN (100 * 1024)

typedef struct WsServer WsServer;

struct WsHandler
{
//    WsServer *server;
//    int socket;
    tcp::socket *sock;
    void *context;
    char resourceName[WS_BUFLEN];
    char buf[WS_BUFLEN];
};

struct WsServer
{
//    int sock;
//    char dirName[80];
//    int port;
    //    pthread_t thread;
    int cont;
    void (*onOpen)(WsHandler* ws, char* msg);
    void (*onClose)(WsHandler* ws, char* msg);
    void (*onMessage)(WsHandler* ws, unsigned char* data, int len);
    void (*onError)(WsHandler* ws, char* msg);
    void* context;
//    WsHandler* clientWs;
};




///**
// *
// */
//int wsSend(WsHandler *ws, char *str);
//
//
///**
// *
// */
//int wsSendBinary(WsHandler *ws, unsigned char *dat, long len);



/**
 *
 */
WsServer *wsCreate(
    void (*onOpen)(WsHandler *, char *),
    void (*onClose)(WsHandler *, char *),
    void (*onMessage)(WsHandler *, unsigned char *, int),
    void (*onError)(WsHandler *, char *),
    void *context, char *dir, int port);



/**
 *
 */
void wsDelete(WsServer *obj);



/**
 *
 */
int wsServe(WsServer *obj);


///**
// * Return an existing client websocket, if any, else NULL
// */
//WsHandler *wsGetClientWs(WsServer *obj);


void sha1hash64(unsigned char* data, int len, char* b64buf);

#if defined(__cplusplus) || defined(c_plusplus)
}  // extern "C"
#endif

#endif  /* _WSSERVER_H_ */
