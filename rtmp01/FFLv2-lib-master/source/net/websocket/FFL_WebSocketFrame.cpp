
#include <net/websocket/FFL_WebSocketFrame.hpp>
#include <net/http/FFL_HttpRequest.hpp>
#include <net/http/FFL_HttpResponse.hpp>
#include <net/FFL_NetEventLoop.hpp>


namespace FFL {
	

	WebsocketFrame::WebsocketFrame() {
			reset();
		}
	WebsocketFrame::~WebsocketFrame() {
		}

		void WebsocketFrame::reset() {
			FIN = false;
			mOpcode = 0;
			mHaveMask = false;
			mPayloadLen = 0;
			mMaskey[0] = 0;
			mMaskey[1] = 0;
			mMaskey[2] = 0;
			mMaskey[3] = 0;
		}

		bool WebsocketFrame::readHeader(NetStreamReader* reader) {
			bool suc = false;
			reader->pull(4000);

			uint8_t b=reader->read1Bytes(&suc);
			if (!suc) {
				return false;
			}
			FIN =(b & 0x80)!=0;
			mOpcode = b & 0x0f;

			b = reader->read1Bytes(&suc);
			if (!suc) {
				return false;
			}
			mHaveMask = (b & 0x80)!=0;
			uint8_t payloadLenFlag= b & 0x7f;
			if (payloadLenFlag <= 125) {
				mPayloadLen = payloadLenFlag;
			}else if (payloadLenFlag == 126) {
				mPayloadLen = reader->read2Bytes(&suc);
			}else if (payloadLenFlag == 127) {
				mPayloadLen = reader->read8Bytes(&suc);
			}

			if (mHaveMask) {
				mMaskey[0] = reader->read1Bytes(&suc);
				mMaskey[1] = reader->read1Bytes(&suc);
				mMaskey[2] = reader->read1Bytes(&suc);
				mMaskey[3] = reader->read1Bytes(&suc);
			}

			return true;
		}
		bool WebsocketFrame::readData(NetStreamReader* reader, uint8_t* buffer, uint32_t* bufferSize) {
			if (mPayloadLen > *bufferSize) {
				return false;
			}

			uint32_t readSize = (uint32_t)mPayloadLen;
			if (!reader->readBytes((int8_t*)buffer, readSize)) {				
				return false;
			}

			if (bufferSize) {
				*bufferSize = readSize;
			}
			return true;
		}
		bool WebsocketFrame::writeHeader(TcpClient* client) {
			uint8_t header[16] = {};
			int32_t headerSize = 2;
			if (FIN) {
				header[0] |= 0x80;
			} else {
				header[0] &= ~0x80;
			}
			header[0] |= mOpcode & 0x0f;

			if (mHaveMask) {
				header[1] = 0x80;
			} else {
				header[1] = 0;
			}

			if (mPayloadLen < 125) {
				header[1] |= mPayloadLen;
			}else if (mPayloadLen < 65536) {
				header[1] |= 126;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
			}
			else if (mPayloadLen < 65536) {
				header[1] |= 127;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
				header[headerSize++] = 0;
			}

			if (mHaveMask) {
				header[headerSize++] = 1;
				header[headerSize++] = 2;
				header[headerSize++] = 3;
				header[headerSize++] = 4;
			}

			
			client->write(header, headerSize, 0);
			return true;
		}
}
