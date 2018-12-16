  cmake_minimum_required(VERSION 3.5)
  project(cmake01 C CXX)

  set(CMAKE_CXX_COMPLICE '/usr/bin/clang++')
#set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)

  set(SOURCE_FILES main.cpp Thread.h)
  add_executable(cmake01 ${SOURCE_FILES})
  target_link_libraries(cmake01)
