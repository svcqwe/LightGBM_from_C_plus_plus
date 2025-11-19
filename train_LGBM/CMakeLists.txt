cmake_minimum_required(VERSION 3.17)
project(train_LGBM)

set(CMAKE_CXX_STANDARD 20)

# Папки с заголовками
include_directories(
        J:/train_LGBM/include
        J:External_c++_libs/LightGBM/include
)

# Папки с библиотеками
link_directories(
        J:/train_LGBM/lib
        J:/External_c++_libs/LightGBM
)

add_executable(train_LGBM
        main.cpp
        )

# Линкуем сразу всё
target_link_libraries(train_LGBM
        pq
        lib_lightgbm.dll     
        )
