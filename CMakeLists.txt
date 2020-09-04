cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
if(WIN32)
	cmake_policy(SET CMP0091 NEW)
endif()
project(epicinium-neuralnewt)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(Torch REQUIRED)

set(CXX_STANDARD 11)
if(WIN32)
	set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
	add_compile_options("/Ox" "/Oi" "/Ot" "/GT" "/GL" "/arch:AVX2" "/fp:fast" "/Zi")
	add_link_options("/LTCG")
else()
	add_compile_options("-pedantic" "-pedantic-errors" "-Wall" "-Wextra" "-g" "-Ofast" "-march=native" "-flto")
endif()

include_directories(${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_SOURCE_DIR})
include_directories(src)
#add_definitions(-DORDERSENCODED)
#add_definitions(-DDEVELOPMENT)

add_library(crypto SHARED IMPORTED)
if(WIN32)
	set_target_properties(crypto PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/openssl/libcrypto-1_1-x64.dll IMPORTED_IMPLIB ${CMAKE_SOURCE_DIR}/libs/openssl/libcrypto.dll.lib)
else()
	set_target_properties(crypto PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/openssl/libcrypto.so)
endif()

add_executable(main libs/jsoncpp/jsoncpp.cpp
                    src/nnet/module.cpp
                    src/nnet/neuralnewtbrain.cpp
                    src/brainname.cpp
                    src/gamedirector.cpp
                    src/newtbraintrainer.cpp
                    src/setting.cpp
                    src/main.cpp)
if(WIN32)
	target_link_libraries(main ${CMAKE_SOURCE_DIR}/libs/aftermath/epicinium-automaton.lib)
else()
	target_link_libraries(main ${CMAKE_SOURCE_DIR}/libs/aftermath/epicinium-automaton.a)
endif()
target_link_libraries(main crypto)
target_link_libraries(main ${TORCH_LIBRARIES})

add_library(neuralnewt EXCLUDE_FROM_ALL SHARED libs/jsoncpp/jsoncpp.cpp
                              src/nnet/module.cpp
                              src/nnet/neuralnewtbrain.cpp
                              src/brainname.cpp
                              src/libneuralnewt.cpp
                              src/setting.cpp)
target_compile_options(neuralnewt PRIVATE "-fvisibility=hidden" "-fvisibility-inlines-hidden")
target_link_options(neuralnewt PRIVATE "-nodefaultlibs" "-ffunction-sections" "-fdata-sections" "-Wl,--gc-sections")
if(WIN32)
	target_link_libraries(neuralnewt ${CMAKE_SOURCE_DIR}/libs/aftermath/epicinium-automaton.lib)
else()
	target_link_libraries(neuralnewt ${CMAKE_SOURCE_DIR}/libs/aftermath/epicinium-automaton.a)
endif()
target_link_libraries(neuralnewt ${TORCH_LIBRARIES})
