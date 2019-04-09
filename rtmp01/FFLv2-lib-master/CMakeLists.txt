cmake_minimum_required(VERSION 3.4.1)

PROJECT(FFL)
#SET(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install) 

INCLUDE(./cmakemode/FFLv2Utils.cmake)

###############################################################################################################################
#生成日志等级宏
# dubug:4 , info:3, warning:2,error:1,crit:0
SET(FFLIB_COMPILE_LOG_LEVEL "3" CACHE PATH "Log Level")
#系统级错误
IF(FFLIB_COMPILE_LOG_LEVEL GREATER 0  OR  FFLIB_COMPILE_LOG_LEVEL EQUAL "0")
  SET(FFL_LOG_CRIT 
     "#define FFL_LOG_CRIT(format,...) FFL_LogPrint(FFL_LOG_LEVEL_CRIT,format,##__VA_ARGS__)\n#define FFL_LOG_CRIT_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_CRIT,tag,format,##__VA_ARGS__)")
ELSE()
  SET(FFL_LOG_CRIT 
    "#define FFL_LOG_CRIT(format,...)\n#define FFL_LOG_CRIT_TAG(tag,format,...) ")  
ENDIF()
#错误等级
IF(FFLIB_COMPILE_LOG_LEVEL GREATER "1"  OR  FFLIB_COMPILE_LOG_LEVEL EQUAL "1")
 SET(FFL_LOG_ERROR 
     "#define FFL_LOG_ERROR(format,...) FFL_LogPrint(FFL_LOG_LEVEL_ERROR,format,##__VA_ARGS__)\n#define FFL_LOG_ERROR_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_ERROR,tag,format,##__VA_ARGS__)")					
ELSE()
  SET(FFL_LOG_ERROR 
     "#define FFL_LOG_ERROR(format,...)\n#define FFL_LOG_ERROR_TAG(tag,format,...) ")  
ENDIF()
#警告级别
IF(FFLIB_COMPILE_LOG_LEVEL GREATER "2"  OR  FFLIB_COMPILE_LOG_LEVEL EQUAL "2")
  SET(FFL_LOG_WARING 
     "#define FFL_LOG_WARNING(format,...) FFL_LogPrint(FFL_LOG_LEVEL_WARNING,format,##__VA_ARGS__)\n#define FFL_LOG_WARNING_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_WARNING,tag,format,##__VA_ARGS__)")					
ELSE()
  SET(FFL_LOG_WARING 
     "#define FFL_LOG_WARNING(format,...)\n#define FFL_LOG_WARNING_TAG(tag,format,...) ")  
ENDIF()

#运行信息
IF(FFLIB_COMPILE_LOG_LEVEL GREATER "3" OR  FFLIB_COMPILE_LOG_LEVEL EQUAL "3")
 SET(FFL_LOG_INFO 
     "#define FFL_LOG_INFO(format,...) FFL_LogPrint(FFL_LOG_LEVEL_INFO,format,##__VA_ARGS__) \n#define FFL_LOG_INFO_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_INFO,tag,format,##__VA_ARGS__)")					
ELSE()
  SET(FFL_LOG_INFO 
      "#define FFL_LOG_INFO(format,...)\n#define FFL_LOG_INFO_TAG(tag,format,...) ")  
ENDIF()


#调试级别
IF(FFLIB_COMPILE_LOG_LEVEL GREATER "4"  OR  FFLIB_COMPILE_LOG_LEVEL EQUAL "4"   )
 SET(FFL_LOG_DEBUG 
     "#define FFL_LOG_DEBUG(format,...) FFL_LogPrint(FFL_LOG_LEVEL_DEBUG,format,##__VA_ARGS__)\n#define FFL_LOG_DEBUG_TAG(tag,format,...) FFL_LogPrintTag(FFL_LOG_LEVEL_DEBUG,tag,format,##__VA_ARGS__)")					
ELSE()
  SET(FFL_LOG_DEBUG 
     "#define FFL_LOG_DEBUG(format,...) \n#define FFL_LOG_DEBUG_TAG(tag,format,...) ")  
ENDIF()

#  生成日志等级宏FFL_LogConfig.h
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/include/FFL_LogConfig.h.in" "${PROJECT_BINARY_DIR}/FFL_LogConfig.h")
###############################################################################################################################
IF(APPLE)
SET(FFLIB_COMPILE_STATIC ON CACHE BOOL "build static library")
#    ios仅仅支持静态库，需要修改CMAKE_SYSROOT路径指向ios sdk路径上
ELSE()
# static lib
SET(FFLIB_COMPILE_STATIC ON CACHE BOOL "build static library")
ENDIF()
#是否生成example
SET(FFLIB_COMPILE_EXAMPLE OFF CACHE BOOL "build static example")

# 编译静态库还是动态库
IF (FFLIB_COMPILE_STATIC)
    set(FFLIB_IMPORT "")
    set(FFLIB_EXPORT "")
ELSE(FFLIB_COMPILE_STATIC)
    if (NOT MSVC)
	   set(FFLIB_IMPORT "")
	   set(FFLIB_EXPORT "__attribute__((visibility(\"default\")))")
    else ()
	   set(FFLIB_IMPORT "__declspec(dllimport)")
	   set(FFLIB_EXPORT "__declspec(dllexport)")
    endif ()
ENDIF (FFLIB_COMPILE_STATIC )

#使用哪一个线程库
IF(WIN32)
OPTION(FFLIB_USING_THREAD_WIN32 "using win32 system thread" ON )	
OPTION(FFLIB_USING_THREAD_STD "using std thread" OFF )
ELSE()
OPTION(FFLIB_USING_THREAD_PTHREAD "using pthread" ON )	
OPTION(FFLIB_USING_THREAD_STD "using std thread" OFF )
ENDIF()

if(FFLIB_USING_THREAD_WIN32)
SET(HAVE_THREAD_WIN "#define FFL_THREAD_WINDOWS 1")
SET(FFLIB_USING_THREAD_STD OFF)
else()
SET(HAVE_THREAD_WIN "#define FFL_THREAD_WINDOWS 0")
endif()


if(FFLIB_USING_THREAD_PTHREAD)
SET(HAVE_THREAD_PTHREAD "#define FFL_THREAD_PTHREAD 1")
else()
SET(HAVE_THREAD_PTHREAD "#define FFL_THREAD_PTHREAD 0")
endif()


if(FFLIB_USING_THREAD_STD)
SET(HAVE_THREAD_STDCPP "#define FFL_THREAD_STDCPP 1")
else()
SET(HAVE_THREAD_STDCPP "#define FFL_THREAD_STDCPP 0")
endif()

#生成配置文件 FFL_Config.h
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/include/FFL_Config.h.in" "${PROJECT_BINARY_DIR}/FFL_Config.h")
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR};include;source)


IF(WIN32)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ")
  SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
ELSE()
#默认的函数隐藏，不导出
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -fvisibility=hidden")
  SET(CMAKE_CXX_FLAGS "-std=c++11 -fvisibility=hidden ${CMAKE_CXX_FLAGS}")
ENDIF()

SET(HEADER_FILES 
   ./include/FFL_lib.h
   ./include/FFL_lib.hpp
   ./include/FFL_Netlib.h
   ./include/FFL_Netlib.hpp
#最基础的功能   
   ./include/FFL_Version.h
   ./include/FFL_Platform.h
   ./include/FFL_Stdint.h
   ./include/FFL_Error.h
   ./include/FFL_Assert.h
   ./include/FFL_Memory.h
   ./include/FFL_String.h
   ./include/FFL_Time.h
   ./include/FFL_Atomic.h
   ./include/FFL_Core.h
   
   ./include/FFL_Log.h
   ./include/FFL_Path.h
   ./include/FFL_Console.h
   ./include/FFL_Console.hpp
   
   ./include/FFL_Thread.h   
   ./include/FFL_ThreadConstant.h
   ./include/FFL_Threadpool.h 
   ./include/FFL_Mutex.h
   
   ./include/FFL_Thread.hpp
   ./include/FFL_Mutex.hpp
   
   ./include/FFL_BlockList.hpp
   ./include/FFL_ByteBuffer.hpp
   ./include/FFL_ByteReader.hpp
   ./include/FFL_ByteStream.hpp
   ./include/FFL_ByteWriter.hpp
   ./include/FFL_Dictionary.hpp
   ./include/FFL_Flags.hpp
   ./include/FFL_Io.hpp      
   ./include/FFL_Module.hpp
   ./include/FFL_Serializable.hpp
   ./include/FFL_String.hpp   
   ./include/FFL_Utils.hpp   
   ./include/FFL_SharedBuffer.hpp   
   ./include/FFL_File.hpp
  
   ./include/FFL_Ref.hpp
   ./include/FFL_RefAtomic.hpp
   ./include/FFL_RefBase.hpp
   ./include/FFL_RefCount.hpp   
   ./include/FFL_RefSp.hpp
   ./include/FFL_RefWeakimpl.hpp
   ./include/FFL_RefWp.hpp

    

   )

   SET(HEADER_NET_FILES 
#网络部分
   ./include/net/FFL_Net.h
   ./include/net/FFL_NetConst.h
      
   ./include/net/FFL_NetEventLoop.hpp
   ./include/net/FFL_NetSocket.hpp
   ./include/net/FFL_NetStream.hpp
   ./include/net/FFL_NetUtils.hpp
   ./include/net/FFL_TcpClient.hpp
   ./include/net/FFL_TcpListener.hpp
   ./include/net/FFL_TcpServer.hpp
   ./include/net/FFL_UdpClient.hpp
   ./include/net/FFL_UdpServer.hpp
   
   )

    SET(HEADER_HTTP_FILES 
#http
    ./include/net/http/FFL_Http.hpp
	./include/net/http/FFL_HttpClient.hpp
	./include/net/http/FFL_HttpClientAccess.hpp
	./include/net/http/FFL_HttpHeader.hpp
	./include/net/http/FFL_HttpParser.hpp
	./include/net/http/FFL_HttpRequest.hpp	
	./include/net/http/FFL_HttpResponse.hpp	
	./include/net/http/FFL_HttpTransportBase.hpp	
	./include/net/http/FFL_HttpFile.hpp
	./include/net/http/FFL_HttpServer.hpp	
	./include/net/http/FFL_HttpUrl.hpp

   )

   SET(HEADER_WEBSOCKET_FILES 
#websocket
    ./include/net/websocket/FFL_WebSocketClient.hpp
	./include/net/websocket/FFL_WebSocketServer.hpp
   )

   
SET(SOURCE_FILES   
  ./source/FFL_Version.c
  ./source/FFL_Error.c
  ./source/FFL_Memory.c
  ./source/FFL_String.c
  ./source/FFL_Time.c
  ./source/FFL_Atomic.c
  ./source/FFL_Core.c
  ./source/FFL_Core.cpp
  
  ./source/FFL_Log.c
  ./source/FFL_Path.c
  
  ./source/FFL_Console.c
  ./source/FFL_Console.cpp  
  ./source/getopt/getopt.h
  ./source/getopt/getopt.c
  ./source/getopt/getopt_int.h
  ./source/getopt/getopt_int.c

  ./source/atomic/atomic.h
  ./source/atomic/atomic.cpp
  
  ./source/thread/FFL_thread.c
  ./source/thread/FFL_threadpool.c
  ./source/thread/threadpool/threadpool.h
  ./source/thread/threadpool/threadpool.c
  ./source/thread/threadpool/threadpool_job.h
  ./source/thread/threadpool/threadpool_job.c
  
  ./source/thread/FFL_Mutex.cpp
  ./source/thread/FFL_Thread.cpp
  ./source/FFL_ByteBuffer.cpp
  ./source/FFL_ByteStream.cpp
  ./source/FFL_Dictionary.cpp
  ./source/FFL_File.cpp
  ./source/FFL_Flags.cpp
  ./source/FFL_Module.cpp
  ./source/FFL_Serializable.cpp
  ./source/FFL_String.cpp
  ./source/FFL_SharedBuffer.cpp  
  ./source/FFL_RefBase.cpp
  ./source/FFL_RefWakeimpl.cpp

  ./source/FFL_Sha1.hpp
  ./source/FFL_Sha1.cpp
  ./source/FFL_Base64.hpp
  ./source/FFL_Base64.cpp  
  
)

IF(FFLIB_USING_THREAD_WIN32)
SET(SOURCE_FILES ${SOURCE_FILES}
  ./source/thread/windows/sysmutex.c
  ./source/thread/windows/systhread.c
  ./source/thread/windows/syssem.c
  ./source/thread/windows/syscond.c
)
ENDIF()

IF(FFLIB_USING_THREAD_STD)
SET(SOURCE_FILES ${SOURCE_FILES}
  ./source/thread/stdcpp/syscond.cpp
  ./source/thread/stdcpp/sysmutex.cpp
  ./source/thread/stdcpp/systhread.cpp
  ./source/thread/stdcpp/syssem.c
)
ENDIF()

IF(FFLIB_USING_THREAD_PTHREAD)
SET(SOURCE_FILES ${SOURCE_FILES}
  ./source/thread/pthread/syscond.c
  ./source/thread/pthread/sysmutex.c
  ./source/thread/pthread/systhread.c
  ./source/thread/pthread/syssem.c
  ./source/thread/pthread/systls.c
)
ENDIF()

#网络部分
AUX_SOURCE_DIRECTORY(./source/net/base/ DIR_SRCS)
SET(SOURCE_FILES ${DIR_SRCS} ${SOURCE_FILES})

AUX_SOURCE_DIRECTORY(./source/net/ DIR_SRCS)
SET(SOURCE_FILES ${DIR_SRCS} ${SOURCE_FILES})

AUX_SOURCE_DIRECTORY(./source/net/http DIR_SRCS)
SET(SOURCE_FILES ${DIR_SRCS} ${SOURCE_FILES})

AUX_SOURCE_DIRECTORY(./source/net/http/http-parser-2.1 DIR_SRCS)
SET(SOURCE_FILES ${DIR_SRCS} ${SOURCE_FILES})

AUX_SOURCE_DIRECTORY(./source/net/websocket DIR_SRCS)
SET(SOURCE_FILES ${DIR_SRCS} ${SOURCE_FILES})


SET(LOCAL_SRC_FILES 
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${HEADER_NET_FILES}
  ${HEADER_HTTP_FILES}
  ${HEADER_WEBSOCKET_FILES})

group_by_dir(${CMAKE_SOURCE_DIR} ${LOCAL_SRC_FILES})

if (MSVC)
    add_definitions("/wd4819 /wd4996")
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endif(MSVC)


#编译对应库文件
if (FFLIB_COMPILE_STATIC)   
   ADD_LIBRARY( FFL STATIC ${LOCAL_SRC_FILES})
else()   
   ADD_LIBRARY( FFL SHARED ${LOCAL_SRC_FILES}) 
endif()
SET_TARGET_PROPERTIES(FFL PROPERTIES CLEAN_DIRECT_OUTPUT 1)

if(FFLIB_COMPILE_EXAMPLE)
ADD_SUBDIRECTORY(example)
endif()

#安装脚本
# 安装${HEADERS}文件到include目录
# 安装生成文件到lib目录
SET(HEADERS 
    ${HEADER_FILES} 
    ${PROJECT_BINARY_DIR}/FFL_Config.h
	${PROJECT_BINARY_DIR}/FFL_LogConfig.h
   )
install(FILES ${HEADERS} DESTINATION include/)
install(FILES ${HEADER_NET_FILES} DESTINATION include/net/)
install(FILES ${HEADER_HTTP_FILES} DESTINATION include/net/http/)
install(FILES ${HEADER_WEBSOCKET_FILES} DESTINATION include/net/websocket/)


install(TARGETS FFL DESTINATION lib/)
