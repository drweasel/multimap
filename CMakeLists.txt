cmake_minimum_required(VERSION 3.10)

project(HashMultiMap LANGUAGES CXX VERSION 1.0.0 DESCRIPTION HashMultiMap)
set(CMAKE_BUILD_TYPE Debug)

find_package(GTest)
enable_testing()

add_compile_options(-std=c++17 -march=native -Wno-ignored-attributes)

set(SOURCES
  hash_multimap.h
  )

add_executable(hash_test hash_test.cpp ${SOURCES})
target_link_libraries(hash_test PRIVATE GTest::GTest GTest::Main)
target_include_directories(hash_test PRIVATE
  "${googletest_ROOT_LOCATION}/include"
  )

# vim:fenc=utf-8 expandtab sw=2:

