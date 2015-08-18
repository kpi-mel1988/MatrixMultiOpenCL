#-------------------------------------------------
#
# Project created by QtCreator 2015-07-25T06:24:19
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = CLMatrixMulti
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

INCLUDEPATH += /usr/local/lib/include


INCLUDEPATH +=  /home/mel/qt/buildroot-2014.08-sk/output/build/opencv-2.4.8/include/opencv

LIBS += -lm  -lpthread -lgobject-2.0  -lgmodule-2.0 -lglib-2.0 -lOpenCL -lGAL


CONFIG += link_prl

PATH += /usr/lib/opencv


target.path = /home/mel/exam
INSTALLS += target
INSTALLS += -lopencv_core

INCLUDEPATH += $$PWD/../../opencvBuild/include
DEPENDPATH += $$PWD/../../opencvBuild/include

unix:!macx: LIBS += -L$$PWD/../../opencvBuild/lib/ -lopencv_core

INCLUDEPATH += $$PWD/../../opencvBuild/include

INCLUDEPATH +=/usr/include/CL

INCLUDEPATH +=/usr/include/GL

DEPENDPATH += $$PWD/../../opencvBuild/include

HEADERS += \
    CLHead.h
