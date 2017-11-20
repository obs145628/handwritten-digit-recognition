cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -Werror -pedantic -std=c++14")
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -Werror -pedantic -std=c++14 -O3 -DNDEBUG")

include_directories(src/)


set(LIB_SRC
  src/date.cc
  src/matrix.cc
  src/neural-network.cc
  src/random.cc
  src/vector.cc
)

set (DATASETS_SRC
  src/mnist-data.cc
)

set(APP_SRC
  ${LIB_SRC}
  ${DATASETS_SRC}

  src/main.cc
)

set(TEST_SRC
  ${LIB_SRC}
  ${DATASETS_SRC}

  tests/main.cc
)


add_executable(app ${APP_SRC})
target_link_libraries(app pthread)

add_executable(test.out ${TEST_SRC})
target_link_libraries(test.out pthread)
add_custom_target(check COMMAND ./test.out)
add_dependencies(check test.out)
