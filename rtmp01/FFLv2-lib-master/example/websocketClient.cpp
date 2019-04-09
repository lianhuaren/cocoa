
#include <FFL_Netlib.hpp>
#include <net/websocket/FFL_WebSocketClient.hpp>

class ClientReader : public FFL::NetEventLoop::Callback {
public:
	//
	//  返回是否还继续读写
	//  priv:透传数据
	//
	virtual bool onNetEvent(NetFD fd, bool readable, bool writeable, bool exception, void* priv) {
		uint8_t buffer[1024] = {};
		uint32_t size = 1024;
		if (mClient->recvFrame(buffer, &size)) {
			printf("recv: %s", buffer);
			return true;
		}
		return false;
	}

	FFL::sp<FFL::WebSocketClient> mClient;
};
int FFL_main() {
	char exePath[1024] = {};
	char exeName[1024] = {};
	FFL_getCurrentProcessPath(exePath, 1023, exeName);

	FFL_socketInit();
	
	FFL::sp<FFL::WebSocketClient> client=new  FFL::WebSocketClient();
	if (!client->connect("127.0.0.1", 8800)) {
		printf("connect failed");
		return 0;
	}
	if (!client->handshark("/")) {
		printf("handshark failed");
		return 0;
	}	

	ClientReader reader;
	reader.mClient = client;

	FFL::NetEventLoop eventLoop(5000);
	eventLoop.addFd(client->getFd(),&reader,NULL,NULL);
	eventLoop.start(new FFL::ModuleThread("readThread"));

	while (1) {
		char ch = getchar();
		if (ch == 10 || ch == 13) {
			continue;
		}

		if (ch == 'q') {
			eventLoop.stop();
			break;
		}

		if (ch == 'a') {
			client->sendText("12234565");
		}
	}
	
	FFL_sleep(1000);
	return 0;
}
