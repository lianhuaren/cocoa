/*
*  This file is part of FFL project.
*
*  The MIT License (MIT)
*  Copyright (C) 2017-2018 zhufeifei All rights reserved.
*
*  FFL_HttpFile.cpp
*  Created by zhufeifei(34008081@qq.com) on 2019/02/19
*  https://github.com/zhenfei2016/FFLv2-lib.git
*
*  http文件请求，应答封装
*/

#include <net/http/FFL_HttpFile.hpp>
#include <FFL_File.hpp>
#include <net/http/FFL_HttpClient.hpp>

namespace FFL {

	//
	//  是否支持这文件
	//
	struct ContentTypeMap {
		const char* ext;
		const char* contentType;
	};

	static bool isSupportFile(String filePath, String& fileType) {
		static ContentTypeMap  sMap[] = {
			{".html","text/html" },
			{ ".js","text/javascript" },
			{ ".js.map","text/javascript.map" },
			{ ".css","text/css" },
			{ ".css.map","text/css.map" },
			{ ".json","text/json" },
			{ ".xml","text/xml" },
			{ ".rss","application / rss + xml" },
			{ ".ico","image/x-icon" },
			{ ".png","image/png" },
			{ ".jpeg","image/jpeg" },
			{ ".jpg","image/jpeg" },
			{ ".gif","image/gif" }
		};

		for (int32_t i = 0; i < FFL_ARRAY_ELEMS(sMap); i++) {
			if (filePath.endWith(sMap[i].ext)) {
				fileType = sMap[i].contentType;
				return true;
			}
		}

		return false;
	}

	class HttpFileContent : public HttpContent {
	public:
		HttpFileContent(const char* file){
			mFile = new File();

			String filePath(file);
			if (FFL_OK == mFile->open(filePath)) {

			}
		}
		~HttpFileContent() {
			mFile->close();
		}
	public:
		bool isOpened() {
			return mFile->isOpened();
		}
		//
		//  获取内容大小
		//
		int32_t getSize() {
			return (int32_t) mFile->getSize();
		}
		//
		//  获取内容
		//
		int32_t read(uint8_t* data, int32_t requestSize, bool* suc) {
			size_t readEd = 0;
			if (mFile->read(data, requestSize, &readEd) == FFL_OK) {			
				if (suc) {
					*suc = true;
				}
			}else{
				if (suc) {
					*suc = false;
				}
			}
			return readEd;
		}
	private:
		File* mFile;
	};
	
	//
	//  创建文件
	//
	static bool createFileFromContent(const char* path, FFL::sp<HttpContent>& content) {
		int32_t fileSize = content->getSize();

		bool ret = false;
		//
		// 写文件
		//
		{
			FFL::String filePath(path);
			File file;
			if (FFL_OK != file.create(filePath)) {
				return false;
			}

#define BLOCK_SIZE 4096
			uint8_t buffer[BLOCK_SIZE] = {};
			int32_t writedNum = 0;
			while (fileSize > writedNum) {
				bool suc = false;
				int32_t readEd = content->read(buffer, BLOCK_SIZE, &suc);
				if (!suc) {
					ret = false;
				}

				if (readEd == 0) {
					FFL_sleep(5);
					continue;
				}

				file.write(buffer, readEd, NULL);
				writedNum += readEd;
			}

			ret = fileSize <= writedNum;
		}
		return ret;
	}



	HttpResponseFile::HttpResponseFile(FFL::sp<HttpClient> client) :
		HttpResponse(client){
	}

	HttpResponseFile::~HttpResponseFile() {
	}

	bool HttpResponseFile::writeFile(const char* filePath) {
		if (filePath == NULL || filePath[0] == 0) {
			return false;
		}
		FFL::sp<HttpFileContent> content=new HttpFileContent(filePath);
		if (!content->isOpened()) {
			return false;			
		}

		String contentType;
		if (!isSupportFile(String(filePath), contentType)) {
			return false;			
		}
		mHeader.setContentType(contentType);
		setContent(content);

		return send();
	}
	//
	//  从response中读取文件
	//
	bool HttpResponseFile::readFile(const char* path) {
		int32_t contentLen=mHeader.getContentLength();
		if (contentLen <= 0) {
			return false;
		}

		FFL::sp<HttpContent> content=readContent();
		if (content.isEmpty()) {
			return false;
		}
		return createFileFromContent(path,content);
	}

	HttpRequestFile::HttpRequestFile(FFL::sp<HttpClient> client) {

	}
	HttpRequestFile::~HttpRequestFile() {

	}

	//
	//  写内容到这个response中
	//
	bool HttpRequestFile::writeFile(const char* filePath) {
		if (filePath == NULL || filePath[0] == 0) {
			return false;
		}
		FFL::sp<HttpFileContent> content = new HttpFileContent(filePath);
		if (!content->isOpened()) {
			return false;
		}

		String contentType;
		if (!isSupportFile(String(filePath), contentType)) {
			return false;
		}
		mHeader.setContentType(contentType);
		setContent(content);

		return send();
	}
	//
	//  从response中读取文件
	//
	bool HttpRequestFile::readFile(const char* path) {
		int32_t contentLen = mHeader.getContentLength();
		if (contentLen <= 0) {
			return false;
		}

		FFL::sp<HttpContent> content = readContent();
		if (content.isEmpty()) {
			return false;
		}
		return createFileFromContent(path, content);
	}
}
