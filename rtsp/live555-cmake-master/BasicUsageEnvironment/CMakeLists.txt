#This should be the first line of the CMakeLists.txt
cmake_minimum_required(VERSION 3.2)

# Poor practice, but very common
project(live555 VERSION 4.7.2)


file(GLOB USER_LIBS_FILE ./*.cpp)

add_library(BasicUsageEnvironment 
        STATIC
        ${USER_LIBS_FILE})

add_definitions("-Wall -g")
add_definitions("-O -DSOCKLEN_T=socklen_t -g -D_LARGEFILE_SOURCE=1 -D_FILE_OFFSET_BITS=64 -Wall -DBSD=1")
#add_compile_options(-O -DSOCKLEN_T=socklen_t -g -D_LARGEFILE_SOURCE=1 -D_FILE_OFFSET_BITS=64 -Wall -DBSD=1)

target_include_directories(BasicUsageEnvironment 
        PUBLIC .
        PUBLIC include
        PUBLIC ../UsageEnvironment/include
        PUBLIC ../groupsock/include)
