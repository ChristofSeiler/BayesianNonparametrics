cmake_minimum_required(VERSION 2.4)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND CMAKE_POLICY)

PROJECT(BayesianNonparametrics)

FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
INCLUDE(${USE_ITK_FILE})
ENDIF(ITK_FOUND)

SET(Boost_INCLUDE_DIRS CACHE PATH "")

# Use the include path and library for Qt that is used by VTK.
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${BayesianNonparametrics_SOURCE_DIR}/../PolyaffineTransformationTrees/
  ${BayesianNonparametrics_SOURCE_DIR}/include/
  ${Boost_INCLUDE_DIRS}
)

SET(BayesianNonparametricsHeaders
)

SET(BayesianNonparametricsSrcs
  ${BayesianNonparametrics_SOURCE_DIR}/src/Main.cxx
)

ADD_EXECUTABLE( BayesianNonparametrics ${BayesianNonparametricsHeaders} ${BayesianNonparametricsSrcs} )
TARGET_LINK_LIBRARIES( BayesianNonparametrics
  ${ITK_LIBRARIES}
)
