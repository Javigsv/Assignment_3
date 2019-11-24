#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int MAX_PARTICLES;
int NUM_ITERATIONS;
int TPB;
float DEC_FACTOR;
float TOLERANCE = 1e-6;

typedef struct {
    float3 position;
    float3 velocity;
} Particle;

__global__ void timestepGPU(Particle* array, int nPart, float dec_fact)
{
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId < nPart)
    {
        array[myId].velocity.x = array[myId].velocity.x * dec_fact;
        array[myId].velocity.y = array[myId].velocity.y * dec_fact;
        array[myId].velocity.z = array[myId].velocity.z * dec_fact;
        array[myId].position.x = array[myId].position.x + array[myId].velocity.x; 
        array[myId].position.y = array[myId].position.y + array[myId].velocity.y; 
        array[myId].position.z = array[myId].position.z + array[myId].velocity.z; 
    }
}

void timestepCPU(Particle* array)
{
    for (int i = 0; i < MAX_PARTICLES; i++)
    {
        array[i].velocity.x = array[i].velocity.x * DEC_FACTOR;
        array[i].velocity.y = array[i].velocity.y * DEC_FACTOR;
        array[i].velocity.z = array[i].velocity.z * DEC_FACTOR;
        array[i].position.x = array[i].position.x + array[i].velocity.x; 
        array[i].position.y = array[i].position.y + array[i].velocity.y; 
        array[i].position.z = array[i].position.z + array[i].velocity.z;
    }
}

int compare(Particle* x, Particle* y)
{
    int value = 1;
    for(int i = 0; i < MAX_PARTICLES && value; i++)
    {
        value = value & (x[i].position.x - y[i].position.x < TOLERANCE); 
        value = value & (x[i].position.y - y[i].position.y < TOLERANCE); 
        value = value & (x[i].position.z - y[i].position.z < TOLERANCE); 
        value = value & (x[i].velocity.x - y[i].velocity.x < TOLERANCE); 
        value = value & (x[i].velocity.y - y[i].velocity.y < TOLERANCE); 
        value = value & (x[i].velocity.z - y[i].velocity.z < TOLERANCE); 
    }
    return value;
}

void initArray(Particle* p)
{
    for(int i = 0; i < MAX_PARTICLES; i++)
    {
        p[i].position.x = (float) rand() / RAND_MAX;
        p[i].position.y = (float) rand() / RAND_MAX;
        p[i].position.z = (float) rand() / RAND_MAX;
        p[i].velocity.x = (float) rand() / RAND_MAX;
        p[i].velocity.y = (float) rand() / RAND_MAX;
        p[i].velocity.z = (float) rand() / RAND_MAX;

    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }

void setParameters(int argc, char** argv) 
{
    switch (argc)
    {
        case 5: DEC_FACTOR = atof(argv[4]);
        case 4: TPB = atoi(argv[3]);
        case 3: NUM_ITERATIONS = atoi(argv[2]);   
        case 2: MAX_PARTICLES = atoi(argv[1]); break;
        default: MAX_PARTICLES = 100000; NUM_ITERATIONS = 100; TPB = 256; DEC_FACTOR = 0.9; break; 
    }
}

int main(int argc, char **argv)
{
    setParameters(argc, argv);
    
    //Input parametres:
    //[1] : Number of particles
    //[2] : Number of iterations
    //[3] : Decreasing factor of velocity (optional)

    double iStart, iElapsCPU, iElapsGPU;
    
    //Initialization of pointers
    Particle* pOriginal = (Particle*) malloc(MAX_PARTICLES * sizeof(Particle));
    initArray(pOriginal);

    Particle* pCPU = (Particle*) malloc(MAX_PARTICLES * sizeof(Particle));
    memcpy(pCPU, pOriginal, MAX_PARTICLES * sizeof(Particle));

    //Particle* pForeign = (Particle*) malloc(MAX_PARTICLES * sizeof(Particle));
    Particle* pForeign; 
    cudaHostAlloc(&pForeign, MAX_PARTICLES * sizeof(Particle), cudaHostAllocDefault);
    memcpy(pForeign, pOriginal, MAX_PARTICLES * sizeof(Particle));

    Particle* pGPU;
    cudaMalloc(&pGPU, MAX_PARTICLES * sizeof(Particle));

    //Computing by CPU
    //printf("Computing by CPU... ");
    iStart = cpuSecond();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        timestepCPU(pCPU);
    }
    iElapsCPU = cpuSecond() - iStart;
    //printf("Done\n");
    
    //Computing by GPU
    //printf("Computing by GPU... ");
    iStart = cpuSecond();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        //Moving data to the device
        cudaMemcpy(pGPU, pForeign, MAX_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);   
        timestepGPU<<<(MAX_PARTICLES + TPB - 1)/TPB, TPB>>>(pGPU, MAX_PARTICLES, DEC_FACTOR);
        cudaMemcpy(pForeign, pGPU, MAX_PARTICLES*sizeof(Particle), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    iElapsGPU = cpuSecond() - iStart;
    //printf("Done\n");

    //Sum up
    printf("\nSize of the array: %d\nTPB: %d\n", MAX_PARTICLES, TPB);
    printf("CPU time: %2f\nGPU time: %2f\n", iElapsCPU, iElapsGPU);


    int comp = compare(pForeign, pCPU);

    if (comp)
    {
        //printf("Both arrays are equal\n");
    }
    else 
    {
        printf("Differences between arrays\n");
    }
    return 0;
}