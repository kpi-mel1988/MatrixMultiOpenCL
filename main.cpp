//************************************************************

// Demo OpenCL application to compute a simple vector addition

// computation between 2 arrays on the GPU

// ************************************************************

#include <stdio.h>

#include <stdlib.h>

#include <CL/cl.h>

#include <sys/time.h>

#include <QCoreApplication>
#include <stdio.h>
#include <iostream>


#include <CLHead.h>




#define DEBUG 1

    struct timeval t1, t2;
    struct timezone tz;
    float deltatime;
    float totaltime = 0.0f;
    unsigned int frames = 0;

int multi( int ** in1, int ** in2, int ** out, int last);
int multiStrass(long int ** in1,long int ** in2,long int ** out, int last);
int divideMatrix(int dim[], int divider, int **outMatr, int *range);
int matrixCLcalculus(clContext *clData, optData * optionData, int i, int j, int k);
int matrixTransform(optData * optionData, int i, int j, int k);
int matrixComposer(int ** D, optData * optionData, clContext *clData,int ind1, int ind2);


clContext clData;
optData optionalData;
// OpenCL source code

const char* OpenCLSource[] = {

"__kernel void VectorAdd(__global int* C, __global int* A,__global int* B, __global int* SIZE12)",

"{",

//"Index of the elements to add \n",

"  int tx = get_global_id(0); \n",
"  int elementA,elementB;   \n",
"  int ty = get_global_id(1); \n",
//"  int counter=0; \n",
" int SIZE = *SIZE12;    \n",
//"some Var for cycle with multiplication \n",

"  int k = 0;   \n",

" int value = 0.0; \n",

 "for(k=0 ; k<SIZE ; ++k) {\n",

    "elementA = A[ty*SIZE + k];  \n",
    "elementB = B[k*SIZE + tx];   \n",
    "value += elementA * elementB;      \n",
 "  } \n",

" C[ty*SIZE + tx] = value;   \n",

"}"

};

// Number of elements in the vectors to be added

#define SIZE 2000



int * A[SIZE];
int * B[SIZE];
//     int * C[SIZE];
int * D[SIZE];



using namespace std;

int main(int argc, char *argv[])
{


    if(DEBUG)
    fprintf(stderr,"Matrix init start");


     int * A[SIZE];
     int * B[SIZE];
//     int * C[SIZE];
     int * D[SIZE];





     srand(time(0));



     int i =0;
     int j=0;
     for (i=0; i<SIZE; ++i)
     {

     A[i] = (int * )malloc(sizeof(int)*SIZE);
     B[i] = (int * )malloc(sizeof(int)*SIZE);
//     C[i] = (int *)malloc(sizeof(int)*SIZE);
     D[i] = (int *)malloc(sizeof(int)*SIZE);



        for (j=0;j<SIZE;++j){

          A[i][j] = rand() % 100;
          B[i][j] = rand() % 100;

//        C[i][j] = 0;
          D[i][j] = 0;
\
          }

     }


     if(DEBUG)
     fprintf(stderr,"Matrix init completed\n");

// additional data for paralell work

          int * matrSize[3];
          int range[3];
          int dim[3]= {SIZE, SIZE,SIZE};
          int divider= 500;

       divideMatrix(dim, divider, matrSize,range);

       optionalData.dim = dim;  // matrices dimensions
       optionalData.divider = divider;
       optionalData.matrSize = matrSize;
       optionalData.range = range;



       //  Size of marrices for convinience


            int SIZE1 = optionalData.divider;//optionData->matrSize[ind1][0];
            int SIZE2 = optionalData.divider;//optionData->matrSize[ind1][ind2];
            int SIZE3 = optionalData.divider;//optionData->matrSize[0][ind2];



//  Array initialisation

       optionalData.inMatr1 = A;
       optionalData.inMatr2 = B;


       optionalData.outArr1 = (int*)malloc(sizeof(int)*divider * divider);
       optionalData.outArr2 = (int*)malloc(sizeof(int)*divider * divider);

    int * HostOutputVector = (int*)malloc(sizeof(int)*divider * divider);




 // Print out last five elements of input matrices

           printf("Matrix  A\n");


           for(i=SIZE-5;i<SIZE;++i){

           for(j=SIZE-5;j<SIZE;j++){

          printf("%d ",A[i][j]);
           }

           printf("\n");

           }


           printf("Matrix  B\n");


           for(i=SIZE-5;i<SIZE;++i){

           for(j=SIZE-5;j<SIZE;j++){

          printf("%d ",B[i][j]);
           }

           printf("\n");

           }


//  OpenCL PROCESSING  PART


     //Get an OpenCL platform

     cl_platform_id cpPlatform;

     clGetPlatformIDs(1, &cpPlatform, NULL);

     // Get a GPU device

    cl_device_id cdDevice;

     clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);

     char cBuffer[1024];

     clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);

     printf("CL_DEVICE_NAME: %s\n", cBuffer);

     clGetDeviceInfo(cdDevice, CL_DRIVER_VERSION, sizeof(cBuffer), &cBuffer, NULL);

     printf("CL_DRIVER_VERSION: %s\n\n", cBuffer);

     // Create a context to run OpenCL enabled GPU

     cl_int error;

     clData.GPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);


     if(error != 0)
      fprintf(stderr,"ERROR:: Unable to create Context. ErrorNo: %d\n",error);



     // Create a command-queue on the GPU device

     cl_command_queue cqCommandQueue = clCreateCommandQueue(clData.GPUContext, cdDevice, 0, NULL);


     // Create OpenCL program with source code

     cl_program OpenCLProgram = clCreateProgramWithSource(clData.GPUContext, 15, OpenCLSource, NULL, NULL);

     // Build the program (OpenCL JIT compilation)

     clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);


     if(DEBUG){

     void * res=malloc(500);
    size_t actVal;

     clGetProgramBuildInfo(OpenCLProgram,cdDevice,CL_PROGRAM_BUILD_LOG,500
                           ,(void *)res,&actVal);


     cout<<"Result LOG of compiled kernel" <<static_cast<char*>(res)<<endl;


      }

     // Create a handle to the compiled OpenCL function (Kernel)

     clData.OpenCLMatrMulti = clCreateKernel(OpenCLProgram, "VectorAdd", NULL);


    if(DEBUG){

     size_t val;


     cl_int errInfo = clGetKernelWorkGroupInfo(clData.OpenCLMatrMulti, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t)*3,

      &val, NULL);



         if(errInfo != 0)
         printf("ERROR:: clGetKernelWorkGroupInfo errorNo: %d\n",errInfo);

        printf("CL_KERNEL_WORK_GROUP_SIZE %d, \n",val);


            }

        // Allocate GPU memory for source vectors AND initialize from CPU memory

        clData.GPUBuffIn1 = clCreateBuffer(clData.GPUContext, CL_MEM_READ_ONLY |

        CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE1 * SIZE2,  optionalData.outArr1 , &error);


        if(error != 0)
         fprintf(stderr,"ERROR:: CreateBuffer1  errorNo: %d\n",error);

        error=0;

        clData.GPUBuffIn2 = clCreateBuffer(clData.GPUContext, CL_MEM_READ_ONLY |

        CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE2 * SIZE3, optionalData.outArr2, &error);


        if(error != 0)
         fprintf(stderr,"ERROR:: CreateBuffer2  errorNo: %d\n",error);



        // Allocate output memory on GPU

        error=0;

        clData.GPUOutputVector = clCreateBuffer(clData.GPUContext, CL_MEM_WRITE_ONLY,

        sizeof(int) * SIZE1 * SIZE3, NULL, &error);

        if(error != 0)
         fprintf(stderr,"ERROR:: CreateBuffer2  errorNo: %d\n",error);


        error=0;

        clData.sizeBuff = clCreateBuffer(clData.GPUContext, CL_MEM_WRITE_ONLY |

        CL_MEM_COPY_HOST_PTR, sizeof(int), (void*)&SIZE2, &error);

        if(error != 0)
         fprintf(stderr,"ERROR:: CreateBuffer small errorNo: %d\n",error);



            // In the next step we associate the GPU memory with the Kernel arguments

            clSetKernelArg(clData.OpenCLMatrMulti, 0, sizeof(cl_mem), (void*)&clData.GPUOutputVector);

            clSetKernelArg(clData.OpenCLMatrMulti, 1, sizeof(cl_mem), (void*)&clData.GPUBuffIn1);

            clSetKernelArg(clData.OpenCLMatrMulti, 2, sizeof(cl_mem), (void*)&clData.GPUBuffIn2);

            clSetKernelArg(clData.OpenCLMatrMulti, 3, sizeof(cl_mem), (void*)&clData.sizeBuff);




        //  Take time

                 gettimeofday(&t2, &tz);
        //        deltatime = (float)(t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6);
               t1 = t2;


 int * HostVector1,*HostVector2;


        for(int ind1 =0;ind1<range[0];ind1++){

            for(int ind2 =0;ind2<range[2];ind2++){

                for(int ind3 =0;ind3<range[1];ind3++){




                 //   int SIZE1 = optionalData.matrSize[ind3][0];
                 //   int SIZE2 = optionalData.matrSize[ind1][ind2];
                 //   int SIZE3 = optionalData.matrSize[0][ind2];


                    if(DEBUG)
                    fprintf(stderr,"Indexes ind1= %d ind2= %d  ind3= %d ",ind1,ind2,ind3);





  //          printf("ind1= %d ind2= %d  ind3= %d ",ind1,ind2,ind3);


        matrixTransform(&optionalData, ind1, ind2,ind3);


//        printf("MAtrixTransform passed");





       clData.HostBuffIn1 = (int *)optionalData.outArr1;
       clData.HostBuffIn2 = (int *)optionalData.outArr2;



        clData.GPUBuffOut = HostOutputVector;
        clData.cqCommandQueue = (cl_command_queue)cqCommandQueue;

    matrixCLcalculus(&clData,&optionalData, ind1, ind2,ind3);


            if(DEBUG)
            printf("MAtrixCalculus passed");



matrixComposer(D, &optionalData, &clData,ind1, ind2);


            }

         }
}






     // Cleanup

     clReleaseKernel(clData.OpenCLMatrMulti);

     clReleaseProgram(OpenCLProgram);

     clReleaseCommandQueue(cqCommandQueue);

     clReleaseContext(clData.GPUContext);

     clReleaseMemObject(clData.GPUBuffIn1);

     clReleaseMemObject(clData.GPUBuffIn2);

     clReleaseMemObject(clData.GPUOutputVector);


//     multi(A,B,C);

//     for( int i =0 ; i < SIZE; i++)

//          printf("[%d + %d = %d]\n",HostVector1[i], HostVector2[i], HostOutputVector[i]);



 /*

     printf("HostVector1  A\n");


     for(i=0;i<SIZE;++i){

     for(j=0;j<SIZE;j++){

    printf("%d ",optionalData.outArr1[i*SIZE+j]);
     }

     printf("\n");

     }




     printf("Matrix  A\n");


     for(i=0;i<SIZE;++i){

     for(j=0;j<SIZE;j++){

    printf("%d ",A[i][j]);
     }

     printf("\n");

     }


     printf("Matrix  B\n");


     for(i=0;i<SIZE;++i){

     for(j=0;j<SIZE;j++){

    printf("%d ",B[i][j]);
     }

     printf("\n");

     }

     printf("HostVector2  B\n");


     for(i=0;i<SIZE;++i){

     for(j=0;j<SIZE;j++){

         printf("%d ",optionalData.outArr2[i*SIZE+j]);
     }

     printf("\n");

     }

     printf("OLOLOLO  Result 11\n");



     for(i=0;i<SIZE;++i){

     for(j=0;j<SIZE;j++){

         printf("%d ",clData.GPUBuffOut[i*SIZE+j]);
   }

     printf("\n");

     }

*/
     gettimeofday(&t2, &tz);
       deltatime = (float)(t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6);


   t1 = t2;

      totaltime = deltatime;

       frames++;

           printf("Time of calculus brutal method  with GPU %1.4f \n", totaltime);



//     multi(A,B,C);


     gettimeofday(&t2, &tz);
        deltatime = (float)(t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6);

    totaltime = deltatime;

        frames++;

            printf("Time of calculus CPU method  %1.4f \n", totaltime);





     printf("Output C matrix\n");



     for(i=SIZE-5;i<SIZE;++i){

     for(j=SIZE-5;j<SIZE;j++){

 //    printf("%d ",C[i][j]);
     }

     printf("\n");

     }

     printf("Output D matrix\n");



     for(i=SIZE-5;i<SIZE;++i){

     for(j=SIZE-5;j<SIZE;j++){

     printf("%d ",D[i][j]);
     }

     printf("\n");

     }


     return 0;

}




int multi( int ** in1, int ** in2, int ** out, int last){
// Brutal matrix calculation function
// last - number of cols and rows to calculate from last till end

int i=0;
int j=0;
int k=0;


    if(last>SIZE)
        last=0;

    for(i=last;i<SIZE;i++){
        for(j=last;j<SIZE;j++){
            for(k=0;k<SIZE;k++){

                out[i][j] += in1[i][k]*in2[k][j];

            }


        }


    }

}



int multiStrass(long int ** in1,long int ** in2,long int ** out, int last){
// Matrix multiplication with Straissen algorithm
// last - number of cols and rows to calculate from last till end

int i=0;
int j=0;
int k=0;
long int M1,M2,M3,M4,M5,M6,M7;
long int C11,C21,C12,C22;
int add;


C11=0;
C21=0;
C12=0;
C22=0;


    if(last>SIZE)
         last=0;


    for(i=last;i<SIZE-1;i=i+2){
        for(j=last;j<SIZE-1;j=j+2){
            for(k=0;k<SIZE-1;k=k+2){

                M1 = (in1[i][k]+in1[i+1][k+1])*(in2[k][j]+in2[k+1][j+1]);
                M2 = (in1[i+1][k]+in1[i+1][k+1])*in2[k][j];
                M3 = in1[i][k]*(in2[k][j+1]-in2[k+1][j+1]);
                M4 = in1[i+1][k+1]*(in2[k+1][j]-in2[k][j]);
                M5 = (in1[i][k]+in1[i][k+1])*in2[k+1][j+1];
                M6 = (in1[i+1][k]-in1[i][k])*(in2[k][j]+in2[k][j+1]);
                M7 = (in1[i][k+1]-in1[i+1][k+1])*(in2[k+1][j]+in2[k+1][j+1]);

                C11 += M1+M4-M5+M7;
                C12 += M3+M5;
                C21 += M2+M4;
                C22 += M1 -M2 +M3+M6;
             }

        out[i][j]=C11;
        out[i+1][j]=C21;
        out[i][j+1]=C12;
        out[i+1][j+1]=C22;
        C11=0;
        C21=0;
        C12=0;
        C22=0;
        }

    }

}



int divideMatrix(int dim[], int divider, int ** outMatr, int *range){
// estimate number of submatrices can get from main input matrices  with given divider

    int MaxSIZE=100;

    outMatr[0] = (int*)malloc(sizeof(int*)*MaxSIZE);
    outMatr[1] = (int*)malloc(sizeof(int*)*MaxSIZE);
    outMatr[2] = (int*)malloc(sizeof(int*)*MaxSIZE);

    range[0]=0;range[1]=0;range[2]=0;

    int maxVal = (dim[0]>dim[1]) ? dim[0]:dim[1];
    int count=0;

    maxVal = (maxVal>dim[2]) ? maxVal:dim[2];

fprintf(stderr,"OLOLO  %d",maxVal);

    for (int i=maxVal;i>0;i=i-divider){

        printf("\n");

          if (dim[0]>divider){
             outMatr[0][count] = divider;
             dim[0] -= divider;
 //      fprintf(stderr,"OLOLO1");
             range[0]++;
             printf("%d  ",outMatr[0][count]);
            }
         else if (dim[0]>0){
                outMatr[0][count] = dim[0];
                dim[0]=0;
                range[0]++;
          printf("%d  ",outMatr[0][count]);
            }

         else printf("\t");


         if (dim[1]>divider){
             outMatr[1][count] = divider;
             dim[1] -= divider;
             range[1]++;
             printf("%d ",outMatr[1][count]);

            }
         else if (dim[1]>0){
             outMatr[1][count] = dim[1];
             dim[1]=0;
             range[1]++;
             printf("%d ",outMatr[1][count]);
            }

         else printf("\t");

        if (dim[2]>divider){
             outMatr[2][count] = divider;
             dim[2] -= divider;
             range[2]++;
             printf("%d ",outMatr[2][count]);
            }
        else if (dim[2]>0){
            outMatr[2][count] = dim[2];
            dim[2]=0;
            range[2]++;
            printf("%d ",outMatr[2][count]);
            }

        else printf("\t");

   count++;

    }

return 0;

}



int matrixComposer(int ** D, optData * optionData, clContext *clData,int ind1, int ind2){


    int SIZE1 = optionData->divider;//optionData->matrSize[ind1][0];
    int SIZE2 = optionData->divider;//optionData->matrSize[ind1][ind2];
    int SIZE3 = optionData->divider;//optionData->matrSize[0][ind2];

    int stepI=0,stepJ=0;

int i,j;

        ind1*=optionData->divider;
        ind2*=optionData->divider;


   for (i=0;i<SIZE1;i++){
       for (j=0;j<SIZE3;j++){
           D[i+ind1][j+ind2] += clData->GPUBuffOut[i*SIZE1+j];
         }
    }

printf("sdfsdf");

return 0;

}



int matrixTransform(optData * optionData, int i, int j, int k){

    int ind1,ind2;
    int SIZE1 = optionData->divider;//optionData->matrSize[ind1][0];
    int SIZE2 = optionData->divider;//optionData->matrSize[ind1][ind2];
    int SIZE3 = optionData->divider;//optionData->matrSize[0][ind2];


    i*=optionData->divider;
    j*=optionData->divider;
    k*=optionData->divider;



    if(DEBUG)
        fprintf(stderr,"Before Matrix Transform");


    for (ind1=0;ind1<SIZE1;ind1++){
        for (ind2=0;ind2<SIZE2;ind2++){
           optionData->outArr1[ind1*SIZE1+ind2] = optionData->inMatr1[i+ind1][k+ind2];
          }
    }





   for (ind1=0;ind1<SIZE2;ind1++){
       for (ind2=0;ind2<SIZE3;ind2++){
           optionData->outArr2[ind1*SIZE2+ind2] = optionData->inMatr2[k+ind1][j+ind2];
         }
    }


   if(DEBUG)
       fprintf(stderr,"After Matrix Transform");

return 0;

}



int matrixCLcalculus(clContext *clData, optData * optionData, int i, int j, int k){


    int SIZE1 = optionData->divider;//optionData->matrSize[ind1][0];
    int SIZE2 = optionData->divider;//optionData->matrSize[ind1][ind2];
    int SIZE3 = optionData->divider;//optionData->matrSize[0][ind2];


    if(DEBUG)
        fprintf(stderr,"Before Kernel Load");






    // Launch the Kernel on the GPU

    cl_int errorEvent;

    cl_event ev1 = clCreateUserEvent(clData->GPUContext, &errorEvent );


    if(errorEvent != 0)
     fprintf(stderr,"ERROR:: clEnqueueWriteBuffer  errorNo: %d\n",errorEvent);


   // Write next submatrix to buffer


    cl_int errorWrite1 = clEnqueueWriteBuffer(clData->cqCommandQueue, (cl_mem) clData->GPUBuffIn1, CL_TRUE, 0,

      SIZE1 * SIZE2 * sizeof(int), clData->HostBuffIn1, 0, NULL, NULL);


    cl_int errorWrite2 = clEnqueueWriteBuffer(clData->cqCommandQueue, (cl_mem) clData->GPUBuffIn2, CL_TRUE, 0,

      SIZE1 * SIZE2 * sizeof(int), clData->HostBuffIn2, 0, NULL, NULL);


    cl_int errorWrite3 = clEnqueueWriteBuffer(clData->cqCommandQueue, (cl_mem) clData->sizeBuff, CL_TRUE, 0,

      sizeof(int), &SIZE1, 0, NULL, NULL);


       if(errorWrite2 != 0)
        fprintf(stderr,"ERROR:: clEnqueueWriteBuffer  errorNo: %d\n",errorWrite2);


     size_t WorkSize[2],localWorkSize[2];

          WorkSize[0] = SIZE1; // two dimensional Range
          WorkSize[1] = SIZE3; // two dimensional Range

// try to experiment with localSize

         localWorkSize[0] = 32;
         localWorkSize[1] = 32;

  cl_int errKern =   clEnqueueNDRangeKernel(clData->cqCommandQueue, clData->OpenCLMatrMulti, 2, NULL,

    WorkSize, NULL, 0, NULL,&ev1);


//  For experiment with localsize use next function

//  cl_int errKern =   clEnqueueNDRangeKernel(clData->cqCommandQueue, clData->OpenCLMatrMulti, 2, NULL,

//    WorkSize, localWorkSize, 0, NULL,&ev1);




    if(errKern != 0)
     fprintf(stderr,"ERROR:: clEnqueueNDRangeKernel  errorNo: %d\n",errKern);
     fprintf(stderr,"sizeof(size_t): %d\n",sizeof(size_t));

       clSetUserEventStatus(ev1, CL_COMPLETE );

     cl_int waitEvent= clWaitForEvents (1, &ev1);


    // Copy the output in GPU memory back to CPU memory

    clEnqueueReadBuffer(clData->cqCommandQueue, (cl_mem) clData->GPUOutputVector, CL_TRUE, 0,

    SIZE1 * SIZE3 * sizeof(int), clData->GPUBuffOut, 0, NULL, NULL);



}
