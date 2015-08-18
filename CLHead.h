#ifndef CLHEAD_H
#define CLHEAD_H

#include <CL/cl.h>
#include "opencv/cv.h"


typedef struct {


    int ** inMatr1;
    int ** inMatr2;
    int * outArr1;
    int * outArr2;
    int * dim;
    int divider;
    int ** matrSize;
    int *range;

}optData;



typedef struct {

 char* OpenCLSource;

 cl_platform_id cpPlatform;

 cl_device_id cdDevice;

 cl_context GPUContext;

 cl_command_queue cqCommandQueue;

 cl_mem GPUBuffIn1;

 cl_mem GPUBuffIn2;

 int * GPUBuffOut;

 int * HostBuffIn1;

 int * HostBuffIn2;

 cl_mem GPUOutputVector;

 cl_program OpenCLProgram;

 cl_kernel OpenCLMatrMulti;

 cl_mem clImageIn;

 cl_mem clImageOut;

 cl_mem sizeBuff;


 cv::Mat * cvImageIn;
 cv::Mat * cvImageOut;

 char cBuffer[1024];

 uchar * bufferImageIn,bufferImageOut;

} clContext;




#endif // CLHEAD_H
