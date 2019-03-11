//
//  CFileDownLoad.hpp
//  curl01
//
//  Created by Mac on 2019/2/21.
//  Copyright © 2019 BaiRuiTechnology. All rights reserved.
//

#ifndef CFileDownLoad_hpp
#define CFileDownLoad_hpp

#include <stdio.h>
#include <curl/curl.h>
#include <json.hpp>
#include <string>

namespace ll {
    


enum DOWNLOAD_RESULT
{
    DOWNLOAD_SUCCESS,
    DOWNLOAD_STOP,
    DOWNLOAD_FAILED,
};
    

class CFileDownLoad {
    static CURL *curl_;
    static bool stop_flag;
    
    static void DownloadingThread(const void *args);
    static int StartDownLoadFile(const std::string fileurl, std::string& local_full_path);
    
    static size_t my_write_func(void *buffer, size_t size, size_t count, void *stream);
    
    static int my_progress_func(void* data,double total, double reciced, double ultotal, double ulnow);
    
public:
    CFileDownLoad();
    ~CFileDownLoad();
    
    
    static int RequestDownload(char const*fileurl, char const*local_path);
    
    static void test();
};

class CommonHelper {
    
    
public:
    static void postNotification(std::string jsonStr, int code);
};
    
    

}

#endif /* CFileDownLoad_hpp */
