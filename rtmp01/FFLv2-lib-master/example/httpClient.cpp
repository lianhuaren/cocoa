
#include <FFL_Netlib.hpp>

class HttpCallback : public FFL::HttpClientAccessManager::Callback {
public:
	//
	//  ����Ӧ��
	//  errorNo :������
	//
	virtual void onResponse(FFL::sp<FFL::HttpResponse> response, int32_t errorNo) {
		if (!response.isEmpty()) {
			FFL::HttpHeader header;
			response->getHeader(header);
			FFL_LOG_DEBUG("response: %d  contentLen=%d", response->getStatusCode(), header.getContentLength());
		}
		else {
			FFL_LOG_DEBUG("response: failed errorNo=%d", errorNo);
		}
	}
};

int FFL_main() {
	char exePath[1024] = {};
	char exeName[1024] = {};
	FFL_getCurrentProcessPath(exePath, 1023, exeName);

	FFL_socketInit();
	FFL::HttpClientAccessManager clientMgr;
	clientMgr.start();

	FFL_LOG_DEBUG("saddsad");
	FFL_LOG_DEBUG("saddsad %d",1);

	{
		FFL::sp<FFL::HttpRequest>  request = new FFL::HttpRequest(NULL);
		FFL::HttpUrl url;
		url.parse(FFL::String("http://127.0.0.1:5000/FFLv2?login"));
		request->setUrl(url);
		clientMgr.post(request, new HttpCallback());
	}

	{
		FFL::sp<FFL::HttpRequest>  request = new FFL::HttpRequest(NULL);
		FFL::HttpUrl url;
		url.parse(FFL::String("http://127.0.0.1:5000/FFLv2?login&asdasd=asd"));
		request->setUrl(url);
		clientMgr.post(request, new HttpCallback());
	}

	while (1) {
		FFL_sleep(2000);
	}

	return 0;
}
