#ifndef _FFL_WEBSOCKET_FRAME_HPP_
#define _FFL_WEBSOCKET_FRAME_HPP_


#include <net/websocket/FFL_WebSocketClient.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/FFL_NetEventLoop.hpp>
#include "FFL_WebSocketHandshark.hpp"

namespace FFL {
	class WebsocketFrame {
	public:
		//
		//  Opcode，操作码
		//	0x0：标识一个中间数据包
		//	0x1：标识一个text类型数据包
		//	0x2：标识一个binary类型数据包
		//	0x3 - 7：保留
		//	0x8：标识一个断开连接类型数据包
		//	0x9：标识一个ping类型数据包
		//	0xA：标识一个pong类型数据包
		//	0xB - F：保留
		//
		enum {
			//
			//  中间切片
			OPCODE_SLICE = 0x0,
			OPCODE_TEXT  = 0x1,
			OPCODE_BINARY = 0x2,
			OPCODE_BYE = 0x08,
			OPCODE_PING = 0x09,
			OPCODE_PONG = 0x0A,
		};
	public:
		WebsocketFrame();
		~WebsocketFrame();

		void reset();

		bool readHeader(NetStreamReader* reader);
		bool readData(NetStreamReader* reader, uint8_t* buffer, uint32_t* bufferSize);

		
		bool writeHeader(TcpClient* client);
	public:
		//
		//  服务端到客户端必须不添加掩码
		//  客户端到服务端必须添加掩码
		//

		//   
		//  第1byte
		//
		//  这个消息的最后一个数据包  1bit
		//
		bool FIN;
		//
		//    保留3bit
		//
		// 消息类型，4bit
		// 0x0：中间数据包，0x1：text类型，
		// 0x2: 二进制类型  0x8：断开连接  
		// 0x9: ping        0xa:pong 
		//
		uint8_t mOpcode;

		//
		//  第2byte
		//
		//
		//  是否添加掩码了 ,1bit  ,
		//  客户端发送来的必须有这个标志
		//  服务端发送的必须没有这个标志
		//
		bool mHaveMask;
		//
		//  0-125 1字节，7bit
		//  126  2字节
		//  127  8字节 
		int64_t  mPayloadLen;
		//
		//  mask key用于跟数据xor
		//
		uint8_t mMaskey[4];
	};
	
}

#endif 