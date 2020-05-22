//
//  CFileDownLoad.cpp
//  curl01
//
//  Created by Mac on 2019/2/21.
//  Copyright © 2019 aaaTechnology. All rights reserved.
//

#include "CFileDownLoad.h"
#import <Foundation/Foundation.h>
#include <thread>
#include <iostream>

#ifdef __OBJC__
namespace ll_ios {

    static NSDictionary *dictionaryWithJsonString(std::string jsonStr);

}
#endif

namespace ll {
    
typedef struct DOWNLOAD_DATA_STRUCT
{
    const char * lpFileUrl;
    const char * lpLocalPath;
}DOWNLOADDATA, *LPDOWNLOADDATA;
    
CURL * CFileDownLoad::curl_ = NULL;
bool CFileDownLoad::stop_flag = false;

CFileDownLoad::CFileDownLoad()
{
}

CFileDownLoad::~CFileDownLoad()
{

}
    
int CFileDownLoad::StartDownLoadFile(const std::string fileurl, std::string& local_full_path)
{
    int ret = 0;

    do {
        curl_global_init(CURL_GLOBAL_ALL);
        curl_ = curl_easy_init();
        
        if (curl_ == NULL) {
            ret = DOWNLOAD_FAILED;
            break;
        }
        
        //char *version = curl_version();
        
        CURLcode res;
        
        FILE *outfile_;
        if ((outfile_ = fopen(local_full_path.c_str(), "wb")) == NULL) {
            ret = DOWNLOAD_FAILED;
            break;
        }
        
        curl_easy_setopt(curl_, CURLOPT_URL, fileurl.c_str());
        const long timeout = 60;
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, timeout);
        curl_easy_setopt(curl_, CURLOPT_RESUME_FROM_LARGE, 0);
        
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA,outfile_);
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, &CFileDownLoad::my_write_func);
        curl_easy_setopt(curl_, CURLOPT_NOPROGRESS, false);
        curl_easy_setopt(curl_, CURLOPT_PROGRESSFUNCTION, &CFileDownLoad::my_progress_func);
        curl_easy_setopt(curl_, CURLOPT_PROGRESSDATA, &timeout);
        
        res = curl_easy_perform(curl_);
        
        if (CURLE_OK == res || CURLE_RANGE_ERROR == res)
        {
            ret = DOWNLOAD_SUCCESS;
        }
        else if (CURLE_ABORTED_BY_CALLBACK == res)
        {
            ret = DOWNLOAD_STOP;
        }
        else
        {
            //        LOG4CXX_ERROR(g_logger, "CFileDownLoad::StartDownLoadFile error : " << curl_easy_strerror(res));
            ret = DOWNLOAD_FAILED;
        }
        
        if(0 !=fclose(outfile_))
        {
            //        LOG4CXX_ERROR(g_logger, "CFileDownLoad::StartDownLoadFile:fclose error : " << ferror(outfile_));
            ret = DOWNLOAD_FAILED;
        }
    } while (0);
    
    if (curl_ != NULL)
    {
        curl_easy_cleanup(curl_);
    }
    curl_global_cleanup();
    
    return ret;
}
    
size_t CFileDownLoad::my_write_func(void *buffer, size_t size, size_t count, void *stream)
{
    size_t writed =  fwrite(buffer, size, count, (FILE *)stream);
    return writed;
}

int CFileDownLoad::my_progress_func(void *data, double total, double received, double ultotal, double ulnow )
{
    if (total != 0 || received != 0)
    {
//        pAutoUpdateEvent->DownLoadProgress(local_file_len_ + total, local_file_len_ + received);
        printf(">>>>>>>>>>total:%.2f,received:%.2f",ultotal,received);
    }
    if (stop_flag)
    {
        return CURL_WRITEFUNC_PAUSE;
    }
    return 0;
}
    
void CFileDownLoad::DownloadingThread(const void *args)
{
    DOWNLOADDATA *lpData = (DOWNLOADDATA *)args;
    int code = 0;
    
    std::string fileurl = lpData->lpFileUrl;
    std::string local_path = lpData->lpLocalPath;
    std::string local_full_path;
    
    if (local_path.back() != '/') {
        local_path += '/';
    }
    std::string filename = fileurl.substr(fileurl.find_last_of("/") + 1, fileurl.length());
    local_full_path = local_path + filename;
    
    code = StartDownLoadFile(fileurl, local_full_path);
    
    nlohmann::json j;
    j["errorcode"] = code;
    j["filepath"] = local_full_path;
    
    CommonHelper::postNotification(j.dump(), code);
    
    delete lpData;
}

int CFileDownLoad::RequestDownload(const char *fileurl, const char *local_path)
{
    DOWNLOADDATA *data = new DOWNLOADDATA();
    data->lpFileUrl = fileurl;
    data->lpLocalPath = local_path;
    std::thread t(
                  &CFileDownLoad::DownloadingThread,data);
    //    t.join();
    t.detach();
    
    return 0;
}

void CFileDownLoad::test()
{
    //    CFileDownLoad*    pFileDownLoad_;
    //
    //    pFileDownLoad_ = new CFileDownLoad;
    //
    //    delete pFileDownLoad_;
    
    NSString *documentPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) objectAtIndex:0];
    
    NSString *fileurl = @"http://facex.anychat.cn:9080/AnyChatFaceX/v1/client/download?path=zipDir/admin/20170720/e1e918a1d3264876b47a51007aeeb55d/e1e918a1d3264876b47a51007aeeb55d.zip";
    
    CFileDownLoad::RequestDownload([fileurl UTF8String], [documentPath UTF8String]);
    
}

void CommonHelper::postNotification(std::string jsonStr, int code)
{
    //>>>>>>>平台判断
#ifdef __OBJC__
    NSDictionary *result = ll_ios::dictionaryWithJsonString(jsonStr);
    
    dispatch_async(dispatch_get_main_queue(), ^{
        
        [[NSNotificationCenter defaultCenter]postNotificationName:@"ANYCHATNOTIFY" object:nil userInfo:result];
        
    });
#endif
    
}
    
}

#ifdef __OBJC__
namespace ll_ios {
    NSDictionary *dictionaryWithJsonString(std::string jsonStr)
    {
        NSString *str = [NSString stringWithCString:jsonStr.c_str() encoding:NSUTF8StringEncoding];
        if (str == nil) {
            return nil;
        }
        
        NSData *jsonData = [str dataUsingEncoding:NSUTF8StringEncoding];
        NSError *err;
        NSDictionary *dic = [NSJSONSerialization JSONObjectWithData:jsonData
                                                            options:NSJSONReadingMutableContainers
                                                              error:&err];
        if(err)
        {
            //  NSLog(@"json解析失败：%@",err);
            return nil;
        }
        return dic;
    }
}

#endif

