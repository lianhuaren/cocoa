cmake_minimum_required(VERSION 3.4.1)

PROJECT(httpServer)

INCLUDE_DIRECTORIES(
            ./			
            ../ffl/include/)

SET(LOCAL_SRC_FILES
   main.cpp;httpServer.cpp
   )


if (MSVC)
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
ADD_EXECUTABLE(httpServer  main.cpp;httpServer.cpp)
TARGET_LINK_LIBRARIES(httpServer FFL)

ADD_EXECUTABLE(httpClient  main.cpp;httpClient.cpp)
TARGET_LINK_LIBRARIES(httpClient FFL)

ADD_EXECUTABLE(websocketClient  main.cpp;websocketClient.cpp)
TARGET_LINK_LIBRARIES(websocketClient FFL)

ADD_EXECUTABLE(websocketServer  main.cpp;websocketServer.cpp)
TARGET_LINK_LIBRARIES(websocketServer FFL)


#ADD_EXECUTABLE(memoryLeak main.cpp;memoryLeak.cpp)
#TARGET_LINK_LIBRARIES(memoryLeak FFL)



