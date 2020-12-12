export PATH=$PATH:~/Downloads/webrtc_win/src/tools/gyp/

#export GYP_GENERATOR_OUTPUT=../build

export GYP_GENERATOR_OUTPUT=.

export GYP_GENERATORS=xcode
export GYP_DEFINES="$GYP_DEFINES OS=mac target_arch=x64 clang=1"
gyp --depth=. system_wrappers.gyp 

https://blog.csdn.net/zoominhao/article/details/48048121