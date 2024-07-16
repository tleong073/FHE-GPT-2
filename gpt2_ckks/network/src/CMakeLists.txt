message("Setting source files")

# Source files in this directory
set(SI_SOURCE_FILES ${SI_SOURCE_FILES}
	${CMAKE_CURRENT_LIST_DIR}/network_util.cpp
	${CMAKE_CURRENT_LIST_DIR}/batchpirclient.cpp
	${CMAKE_CURRENT_LIST_DIR}/batchpirparams.cpp
	${CMAKE_CURRENT_LIST_DIR}/batchpirserver.cpp
	${CMAKE_CURRENT_LIST_DIR}/client.cpp
	${CMAKE_CURRENT_LIST_DIR}/main.cpp
	${CMAKE_CURRENT_LIST_DIR}/pirparams.cpp
	${CMAKE_CURRENT_LIST_DIR}/server.cpp
)

set(SI_SOURCE_FILES ${SI_SOURCE_FILES} PARENT_SCOPE)