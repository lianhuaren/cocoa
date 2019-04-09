
#include <FFL_Netlib.hpp>
#include <FFL_Console.h>

volatile int gQuitFlag = 0;
static void stop(const char* args, void* userdata) {
	FFL::HttpServer* httpServer = (FFL::HttpServer*) userdata;
	httpServer->stop();
	gQuitFlag = 1;
}
static CmdOption  myCmdOption[] = {	
	{ "stop",0,stop,"stop server " },
	{ 0,0,0,0 }
};
int QuitFlag(void* userdata) {
	return gQuitFlag;
}


FFL::String gLastLog;
class HttGetLogListHandelr : public FFL::HttpServer::Callback {
public:
	virtual bool onHttpQuery(FFL::HttpRequest* request) {
		FFL::HttpUrl url;
		request->getUrl(url);

		if (0) {
			FFL::sp<FFL::HttpResponse> response = request->makeResponse();
			response->setContent((uint8_t*)(gLastLog.string()), gLastLog.size());
			response->setStatusCode(200);
			response->send();
			response->finish();
		}
		else {
			FFL::sp<FFL::HttpResponseFile> response = 
				new FFL::HttpResponseFile(request->getHttpClient());
			response->setStatusCode(200);
			response->writeFile("d://test.jpg");
			response->finish();
		}
		return false;
	};
};


class HttpApiLoginHandelr : public FFL::HttpServer::Callback {
public:
	virtual bool onHttpQuery(FFL::HttpRequest* request) {		
		FFL::HttpUrl url;
		request->getUrl(url);
		
		FFL::HttpHeader header;
		request->getHeader(header);

		FFL::sp<FFL::HttpContent> content =request->readContent();
		uint8_t buffer[4096] = {};
	    size_t bufSize = 4096;
		size_t readed = 0;
		int32_t requestSize = header.getContentLength();
		FFL_ASSERT((size_t)requestSize <= bufSize);

		if ((readed=content->read(buffer, requestSize,NULL))>0) {
			buffer[readed] = 0;
			FFL_LOG_WARNING_TAG("http:", " requestSize=%d readed=%d", requestSize,readed);
			FFL_LOG_WARNING_TAG("http:", " %s", ((const char*)buffer));
			gLastLog.setTo((const char*)buffer, readed);

		}
		else {
			FFL_LOG_WARNING_TAG("http", "read fail");
		}
		
		FFL::sp<FFL::HttpResponse> response=request->makeResponse();
		response->setStatusCode(200);
		response->send();
		response->finish();

		//FFL::HttpResponseBuilder builder;
		//FFL::sp<FFL::HttpResponseFile> response =builder.createResponseFile(request);

		//FFL::File file;
		//file.open(FFL::String("d://test.txt"));	

		//FFL::HttpHeader header;
		//header.setContentLength(file.getSize());
		//response->setHeader(header);
		//response->guessFileType("test.txt");
		//response->setReader(&file);
		//response->send();			
		return false;
	};
};

class TcpHandler : public FFL::TcpServer::Callback {
public:
	//
	//  aliveTimeUs:保活时长，如果超过这么长时间还没有数据则干掉这个client
	//              <0 一直存活， 
	//
	virtual bool onClientCreate(FFL::TcpClient* client, int64_t* aliveTimeUs) {
		FFL_LOG_DEBUG("TcpHandler:onClientCreate (%p)  ", client);
		return true;
	}
	virtual void onClientDestroy(FFL::TcpClient* client, FFL::TcpServer::Callback::FD_OPTMODE reason) {
		FFL_LOG_DEBUG("TcpHandler:onClientDestroy (%p)  %s ", client);
	}
	virtual FFL::TcpServer::Callback::FD_OPTMODE onClientReceived(FFL::TcpClient* client) {
		size_t readed = 0;
		char buf[4096] = {};
		if (client->read((uint8_t*)buf, 4095, &readed)!=FFL_OK) {
			return FFL::TcpServer::Callback::FD_DESTROY;
		}
		FFL_LOG_DEBUG("TcpHandler:onClientReceived (%p) (len=%d)  %s ",client,readed,buf);
		return FFL::TcpServer::Callback::FD_CONTINUE;
	}
};
int FFL_main() {
	char exePath[1024] = {};
	char exeName[1024] = {};
	FFL_getCurrentProcessPath(exePath, 1023, exeName);

	gLastLog = "hi:";
	gLastLog += exePath;

	//if (0) {
	//	TcpHandler handler;
	//	FFL::TcpServer tcpServer("127.0.0.1",5000,&handler);
	//	tcpServer.start(NULL);
	//	int32_t waitMs = 5000;
	//	while (tcpServer.eventLoop(&waitMs)) {
	//	}
	//}
	//else {

	FFL::sp<HttpApiLoginHandelr> handler = new HttpApiLoginHandelr();
	FFL::HttpServer httpServer("127.0.0.1", 5000);
	
	//
	//  上传日志接口
	//
	FFL::HttpServer::HttpApiKey key;
	key.mPath = "/fflog";		
	httpServer.registerApi(key,handler);

	//
	//  下载日志接口
	//		
	FFL::sp<HttGetLogListHandelr> getListHandler = new HttGetLogListHandelr();
	key.mPath = "/FFLv2";
	key.mQuery = "getLogList";
	httpServer.registerApi(key, getListHandler);

	httpServer.start(new FFL::ModuleThread("httpd"));	
	//
	//  启动命令行
	//
	FFL_consoleEvenLoop(myCmdOption, &httpServer, QuitFlag);

	return 0;
}
