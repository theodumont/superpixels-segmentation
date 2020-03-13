cmake_minimum_required(VERSION 2.6)


set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS_RELEASE}")

# PROJET #

project(boost_tutorial)
set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})

# LIBRAIRIES #

# Boost Python
include_directories("/usr/include/python2.7")

# CODE SOURCE #

#Generates a list of the source files
file(
	GLOB_RECURSE
	source_files
	src/*
)

# LIBRAIRIE #

add_library(
	fmm
	SHARED
	${source_files}
)


# EDITEUR DE LIEN #

target_link_libraries(
	fmm
	boost_python
)


