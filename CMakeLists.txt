cmake_minimum_required(VERSION 3.0)
project(HandwrittenDigitRecognition)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -Werror -std=c++17")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -std=c++17 -O2 -DNDEBUG")

add_subdirectory(ext/ai-cpp/src)
include_directories(ext/ai-cpp/include)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")
find_package(SFML REQUIRED system window graphics network audio)
include_directories(${SFML_INCLUDE_DIR})

set(DNN_SRC
  src/main-dnn.cc
)
add_executable(dnn ${DNN_SRC})
add_dependencies(dnn ai-cpp)
target_link_libraries(dnn ai-cpp ${SFML_LIBRARIES})


set(LOGREG_SRC
  src/main-logreg.cc
)
add_executable(logreg ${LOGREG_SRC})
add_dependencies(logreg ai-cpp)
target_link_libraries(logreg ai-cpp ${SFML_LIBRARIES})


set(SOFTMAXREG_SRC
  src/main-softmaxreg.cc
)
add_executable(softmaxreg ${SOFTMAXREG_SRC})
add_dependencies(softmaxreg ai-cpp)
target_link_libraries(softmaxreg ai-cpp ${SFML_LIBRARIES})

add_custom_target(gen_mnist
	COMMAND
	PYL_BUILD_DIR=${CMAKE_BINARY_DIR}
	PYL_ROOT_DIR=${CMAKE_SOURCE_DIR}
	python ${CMAKE_SOURCE_DIR}/src/gen_mnist.py
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_custom_target(run_dnn
	COMMAND ${CMAKE_BINARY_DIR}/dnn
	DEPENDS dnn gen_mnist
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR})


add_custom_target(run_logreg
	COMMAND ${CMAKE_BINARY_DIR}/logreg
	DEPENDS logreg gen_mnist
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_custom_target(run_softmaxreg
	COMMAND ${CMAKE_BINARY_DIR}/softmaxreg
	DEPENDS softmaxreg gen_mnist
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
