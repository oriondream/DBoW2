project(dbow2)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

PROTOBUF_CATKIN_GENERATE_CPP(PROTO_SRCS PROTO_HDRS proto/brief_node.proto
                                                   proto/brief_vocabulary.proto
                                                   proto/brief_word.proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(SRCS 
  src/BowVector.cpp     src/FBrief.cpp        src/FSurf64.cpp       src/FORB.cpp
  src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp
  src/TemplatedVocabulary.cpp)

add_definitions(-std=c++11 -O3)

cs_add_library(${PROJECT_NAME} ${SRCS} ${PROTO_SRCS})
target_link_libraries(${PROJECT_NAME} opencv_nonfree)

cs_add_executable(voc_to_proto src/voc_to_proto.cpp)
target_link_libraries(voc_to_proto ${PROJECT_NAME})

cs_add_executable(demo demo/demo.cpp)
target_link_libraries(demo ${PROJECT_NAME})
file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)

cs_install()
# we need to add the current binary dir to include dirs due to protobuf
cs_export(INCLUDE_DIRS include ${CMAKE_CURRENT_BINARY_DIR})
