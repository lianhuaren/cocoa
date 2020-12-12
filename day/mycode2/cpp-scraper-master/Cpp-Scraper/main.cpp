//
//  main.cpp
//
//  Created by Daniel Marchena Parreira on 2018-08-25.
//  Copyright © 2018 Daniel Marchena Parreira. All rights reserved.
//
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>

#include <stdio.h>
#include <curl/curl.h>
#include <map>
#include "json.hpp"

std::ofstream dataFile;
bool failedToFetch = false;

size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

void dump(const char *text,
          FILE *stream, unsigned char *ptr, size_t size)
{
    size_t i;
    size_t c;
    unsigned int width=0x10;
    
    fprintf(stream, "%s, %10.10ld bytes (0x%8.8lx)\n",
            text, (long)size, (long)size);
    
    for(i=0; i<size; i+= width) {
        fprintf(stream, "%4.4lx: ", (long)i);
        
        /* show hex to the left */
        for(c = 0; c < width; c++) {
            if(i+c < size)
                fprintf(stream, "%02x ", ptr[i+c]);
            else
                fputs("   ", stream);
        }
        
        /* show data on the right */
        for(c = 0; (c < width) && (i+c < size); c++) {
            char x = (ptr[i+c] >= 0x20 && ptr[i+c] < 0x80) ? ptr[i+c] : '.';
            fputc(x, stream);
        }
        
        fputc('\n', stream); /* newline */
    }
}

int my_trace(CURL *handle, curl_infotype type,
             char *data, size_t size,
             void *userp)
{
    const char *text;
    (void)handle; /* prevent compiler warning */
    (void)userp;
    
    switch (type) {
        case CURLINFO_TEXT:
            fprintf(stderr, "== Info: %s", data);
        default: /* in case a new one is introduced to shock us */
            return 0;
            
        case CURLINFO_HEADER_OUT:
            text = "=> Send header";
            break;
        case CURLINFO_DATA_OUT:
            text = "=> Send data";
            break;
        case CURLINFO_SSL_DATA_OUT:
            text = "=> Send SSL data";
            break;
        case CURLINFO_HEADER_IN:
            text = "<= Recv header";
            break;
        case CURLINFO_DATA_IN:
            text = "<= Recv data";
            break;
        case CURLINFO_SSL_DATA_IN:
            text = "<= Recv SSL data";
            break;
    }
    
    dump(text, stderr, (unsigned char *)data, size);
    return 0;
}

void postTask(std::int32_t pages) {
    
    CURL *curl;
    CURLcode res;
    std::string readBuffer;
    std::map <std::string, std::string> postParameters;
    
    /* In windows, this will init the winsock stuff */
    curl_global_init(CURL_GLOBAL_ALL);
    
    /* get a curl handle */
    curl = curl_easy_init();
    if(curl) {
        /* First set the URL that is about to receive our POST. This URL can
         just as well be a https:// URL if that is what should receive the
         data. */
        curl_easy_setopt(curl, CURLOPT_URL, "http://119.29.108.104:8080/inweb01/aa.json");
        
        curl_easy_setopt(curl, CURLOPT_DEBUGFUNCTION, my_trace);
        
        /* the DEBUGFUNCTION has no effect until we enable VERBOSE */
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
        
        /* Specify the request callback */
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        /* specify the POST data */
        postParameters.emplace("CultureId", "1");
        postParameters.emplace("ApplicationId", "1");
        postParameters.emplace("RecordsPerPage", "9");
        postParameters.emplace("MaximumResults", "9");
        postParameters.emplace("PropertySearchTypeId", "1");
        postParameters.emplace("TransactionTypeId", "2");
        postParameters.emplace("StoreyRange", "0-0");
        postParameters.emplace("BedRange", "0-0");
        postParameters.emplace("BathRange", "0-0");
        postParameters.emplace("LongitudeMin", "-79.41233270339353");
        postParameters.emplace("LongitudeMax", "-79.35413949660642");
        postParameters.emplace("LatitudeMin", "43.62963286426394");
        postParameters.emplace("LatitudeMax", "43.663605932185064");
        postParameters.emplace("SortOrder", "A");
        postParameters.emplace("SortBy", "1");
        postParameters.emplace("viewState", "m");
        postParameters.emplace("Longitude", "-79.38323609999999");
        postParameters.emplace("Latitude", "43.6466218");
        postParameters.emplace("CurrentPage", std::to_string(pages));
        postParameters.emplace("ZoomLevel", "14");
        postParameters.emplace("PropertyTypeGroupID", "1");
        postParameters.emplace("Token", "D6TmfZprLI9Kv5JtoopAH67oXvr3z9weWBbZ0qXajHA=");
        postParameters.emplace("GUID", "4483cfc2-fc47-44d5-80c4-5eca6c173ab1");
        postParameters.emplace("Version", "6.0");
        
        std::map <std::string, std::string> :: iterator itr;
        std::stringstream concatenatePostParams{};
        
        for (itr = postParameters.begin(); itr != postParameters.end(); ++itr)
        {
            concatenatePostParams << itr->first <<  '=' << itr->second <<  '&';
        }
        
        std::string paramStr = concatenatePostParams.str();
        
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, paramStr.c_str());
        
        /* Perform the request, res will get the return code */
        res = curl_easy_perform(curl);
        
        nlohmann::json jsonResponse = nlohmann::json::parse(readBuffer);
        
        /* Check for errors */
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n",
                    curl_easy_strerror(res));
            failedToFetch = true;
        }
//        std::cout << "Response: " << jsonResponse["ErrorCode"]["Id"] << std::endl;
//        if(jsonResponse["ErrorCode"]["Id"] != 200) {
//            std::cout << "curl_easy_perform() failed: " << jsonResponse["ErrorCode"]["Description"] << std::endl;
//            failedToFetch = true;
//        }
        
        dataFile << "Page: " << pages << " c: " << jsonResponse["c"] << std::endl;
    }
    
    /* always cleanup */
    curl_easy_cleanup(curl);
    
    curl_global_cleanup();
}

int main(void)
{
    
    dataFile.open("data.txt");
    std::int32_t pages = 1;
    
    do {
        
        if (failedToFetch) break;
 
        std::thread t1(postTask, pages++);
        t1.join();
        
    } while(pages <=10);
    
    
    dataFile.close();
    
    return 0;
}
