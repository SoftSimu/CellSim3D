#ifndef STD_INC
  #include <stdio.h>
  #include <stdlib.h>
  #include <time.h>
  #include <math.h>
#endif


#ifndef CUDA_INC
  #include <cuda.h>
#endif

#ifndef CELLDIV_INC
  #include "postscript.h"
  #include "marsaglia.h"
#endif


#define MaxNoofC180s 250000

float mass;                                        //  M
float repulsion_range,    attraction_range;        //  LL1, LL2
float repulsion_strength, attraction_strength;     //  ST1, ST2
float Youngs_mod;
float viscotic_damping, internal_damping;          //  C, DMP
int   Division_step, Time_steps;
float delta_t;
int   Restart;
int   trajWriteInt; // trajectory write interval
int   countOnlyInternal; // 0 - Count all new cells
                         // 1 - Count only the cells born within 0.6Rmax from
                         //     the center of mass of the system
float radFrac;
// equilibrium length of springs between fullerene atoms
float R0  = 0.13517879937327418f;

float L1  = 3.0f;       // the initial fullerenes are placed in 
// an X x Y grid of size L1 x L1


// the three nearest neighbours of C180 atoms 
int   C180_nn[3*192];
int   C180_sign[180];
// device: the three nearest neighbours of C180 atoms 
int   *d_C180_nn;
int   *d_C180_sign;

int   CCI[2][271];       // list of nearest neighbor carbon pairs in the fullerne 
// number of pairs = 270

int   C180_56[92*7];     // 12 lists of atoms forming pentagons 1 2 3 4 5 1 1 and
// 80 lists of atoms forming hexagons  1 2 3 4 5 6 1
int   *d_C180_56;  

float *d_volume;
float *volume;


int No_of_threads;
int Side_length;
int ex, ey;


float  *X,  *Y,  *Z;     // host: atom positions

float *d_XP, *d_YP, *d_ZP;     // device: time propagated atom positions
float  *d_X,  *d_Y,  *d_Z;     // device: present atom positions
float *d_XM, *d_YM, *d_ZM;     // device: previous atom positions 

// host: minimal bounding box for fullerene
float *bounding_xyz;     // minx = bounding_xyz[fullerene_no*6+0]
// maxx = bounding_xyz[fullerene_no*6+1]
// miny = bounding_xyz[fullerene_no*6+2]
// maxy = bounding_xyz[fullerene_no*6+3]
// minz = bounding_xyz[fullerene_no*6+4]
// maxz = bounding_xyz[fullerene_no*6+5]

float *d_bounding_xyz;   // device:  bounding_xyz

// global minimum and maximum of x and y, preprocessfirst 
// global minimum and maximum of x and y, postprocesssecond 
float *d_Minx, *d_Maxx, *d_Miny, *d_Maxy, *d_Minz, *d_Maxz;
float *Minx, *Maxx, *Miny, *Maxy, *Minz, *Maxz;
 
float DL;
int Xdiv, Ydiv, Zdiv;

int *d_NoofNNlist;
int *d_NNlist;
int *NoofNNlist;
int *NNlist;

float *d_CMx, *d_CMy, *d_CMz;
float *CMx, *CMy, *CMz;
  
float Pressure;          // pressure
float Temperature;       // equation of state relates Pressure and Temperature
  
int  No_of_C180s;        // the global number of C180 fullerenes             

float *ran2;             // host: ran2[]
float *d_ran2;           // device: ran2[], used in celldivision

int *NDIV;               // # of divisions
  
long int GPUMemory;
long int CPUMemory;


int main(int argc, char *argv[])
{
  int i;
  int rank,globalrank,step;
  int noofblocks, threadsperblock, prevnoofblocks;
  int Orig_No_of_C180s, temp_No_of_C180s, newcells, newcells_imp, No_of_C180s_imp; 
  int reductionblocks;
  float PSS;
  float s, theta, phi;
  FILE *outfile;
  FILE *trajfile; // pointer to xyz file
  cudaError_t myError;

  int* dividingCells; //Cells that are about to divide
  int* totalCells; // No. of cells at every Dividing_steps
  int* newCellInds; 

  float CoMx, CoMy, CoMz, Rmax; // CoM of the whole system, and the max radius from the Com
  float tempX, tempY, tempZ; // multipurpose temporary coords or vectors

  //int min_no_of_cells = 10; 

  printf("CellDiv version 0.9\n"); 

  if ( argc != 2 ) 
	{
	  printf("Usage: CellDiv no_of_threads\n"); 
	  return(0);
	}

  No_of_threads = atoi(argv[1]);


  Side_length   = (int)( sqrt( (double)No_of_threads )+0.5);
  if ( No_of_threads > MaxNoofC180s || Side_length*Side_length != No_of_threads ) 
	{
	  printf("Usage: Celldiv no_of_threads\n"); 
	  printf("       no_of_threads should be a square, n^2, < %d\n", MaxNoofC180s);
	  return(0);
	}

  No_of_C180s      = No_of_threads;                
  Orig_No_of_C180s = No_of_C180s;                
  GPUMemory = 0L;
  CPUMemory = 0L;
  
  if ( read_global_params()               != 0 ) return(-1);
  if ( read_fullerene_nn()                != 0 ) return(-1);
  if ( generate_random(Orig_No_of_C180s)  != 0 ) return(-1);
  if ( initialize_C180s(Orig_No_of_C180s) != 0 ) return(-1);
  NDIV = (int *)calloc(MaxNoofC180s,sizeof(int));
  CPUMemory += MaxNoofC180s*sizeof(int);
  for ( i = 0; i < No_of_threads; ++i ) NDIV[i] = 1;
  for ( i = No_of_threads; i < MaxNoofC180s; ++i ) NDIV[i] = 0;

  PSS         = 0.0f;
  Pressure    = 0.0f;
  Temperature = 0.0f;

  // empty the psfil from previous results
  outfile = fopen("psfil","w");
  if ( outfile == NULL ) {printf("Unable to open file psfil\n");return(-1);}
  fclose(outfile);

  /* PM
	 Allocate memory for the dividingCells array that will be used to
	 calculate the mitotic index.
  */

  dividingCells = (int *)calloc((Time_steps/Division_step) + 1, sizeof(int));
  totalCells = (int *)calloc((Time_steps/Division_step) + 1, sizeof(int));
  
  CPUMemory += (long)(Time_steps/Division_step) + 1L; 
  /*
  printf ("\nTime_steps %d Division_step %d %d\n%d %d %d", Time_steps, Division_step, Time_steps/Division_step + 1, dividingCells[0], dividingCells[1], dividingCells[2]);
  char c; 
  scanf("c", c);
  */
  
  
  getDevice();
  
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_nn, 3*192*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_sign, 180*sizeof(int))) return(-1);
  GPUMemory +=  3*192*sizeof(int) + 180*sizeof(int);
  //  cudaError_t myError = cudaGetLastError();
  //     if ( cudaSuccess != myError )
  //         { printf( "1: Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);}

  if ( cudaSuccess != cudaMalloc( (void **)&d_XP , 192*MaxNoofC180s*sizeof(float))) return(-1); 
  if ( cudaSuccess != cudaMalloc( (void **)&d_YP , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ZP , 192*MaxNoofC180s*sizeof(float))) return(-1);  
  if ( cudaSuccess != cudaMalloc( (void **)&d_X  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Y  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Z  , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_XM , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_YM , 192*MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ZM , 192*MaxNoofC180s*sizeof(float))) return(-1);
  GPUMemory += 9L*192L*MaxNoofC180s*sizeof(float);

  if ( cudaSuccess != cudaMalloc( (void **)&d_bounding_xyz , MaxNoofC180s*6*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMx ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMy ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_CMz ,          MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_volume ,       MaxNoofC180s*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Minx ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxx ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Miny ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxy ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Minz ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_Maxz ,         1024*sizeof(float))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_NoofNNlist ,   1024*1024*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_NNlist ,    32*1024*1024*sizeof(int))) return(-1);
  if ( cudaSuccess != cudaMalloc( (void **)&d_C180_56,       92*7*sizeof(int))) return(-1);

  GPUMemory += MaxNoofC180s*6L*sizeof(float);
  GPUMemory += MaxNoofC180s*3L*sizeof(float);
  GPUMemory += MaxNoofC180s*sizeof(float);
  GPUMemory += 6L*1024L*sizeof(float);
  GPUMemory += 1024L*1024L*sizeof(int);
  GPUMemory += 32L*1024L*1024L*sizeof(int);
  GPUMemory += 92L*7L*sizeof(int);
  if ( cudaSuccess != cudaMalloc( (void **)&d_ran2 , 10000*sizeof(float))) return(-1);
  GPUMemory += 10000L*sizeof(float);

  bounding_xyz = (float *)calloc(MaxNoofC180s*6, sizeof(float));
  CMx   = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMy   = (float *)calloc(MaxNoofC180s, sizeof(float));
  CMz   = (float *)calloc(MaxNoofC180s, sizeof(float));
  volume= (float *)calloc(MaxNoofC180s, sizeof(float));
  Minx  = (float *)calloc(1024, sizeof(float));
  Maxx  = (float *)calloc(1024, sizeof(float));
  Miny  = (float *)calloc(1024, sizeof(float));
  Maxy  = (float *)calloc(1024, sizeof(float));
  Minz  = (float *)calloc(1024, sizeof(float));
  Maxz  = (float *)calloc(1024, sizeof(float));
  NoofNNlist = (int *)calloc( 1024*1024,sizeof(int));
  NNlist =  (int *)calloc(32*1024*1024, sizeof(int));

  CPUMemory += MaxNoofC180s*6L*sizeof(float);
  CPUMemory += MaxNoofC180s*sizeof(float);
  CPUMemory += 3L*MaxNoofC180s*sizeof(float);
  CPUMemory += 6L*1024L*sizeof(float);

  printf("   Total amount of GPU memory used =    %8.2lf MB\n",GPUMemory/1000000.0);
  printf("   Total amount of CPU memory used =    %8.2lf MB\n",CPUMemory/1000000.0);

  cudaMemcpy(d_C180_nn,   C180_nn,   3*192*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_sign, C180_sign, 180*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_C180_56,   C180_56,   7*92*sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcpy(d_XP, X, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_YP, Y, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZP, Z, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_X,  X, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y,  Y, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Z,  Z, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_XM, X, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_YM, Y, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ZM, Z, 192*MaxNoofC180s*sizeof(float),cudaMemcpyHostToDevice);


  prevnoofblocks  = No_of_C180s;
  noofblocks      = No_of_C180s;
  threadsperblock = 192;
  printf("   no of blocks = %d, threadsperblock = %d, no of threads = %ld\n",
		 noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));

  bounding_boxes<<<No_of_C180s,32>>>(No_of_C180s,d_XP,d_YP,d_ZP,d_X,d_Y,d_Z,d_XM,d_YM,d_ZM, 
									 d_bounding_xyz, d_CMx, d_CMy, d_CMz);

  
  reductionblocks = (No_of_C180s-1)/1024+1;
  minmaxpre<<<reductionblocks,1024>>>( No_of_C180s, d_bounding_xyz, 
									   d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
  minmaxpost<<<1,1024>>>(reductionblocks, d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
  cudaMemset(d_NoofNNlist, 0, 1024*1024);
  cudaMemcpy(Minx, d_Minx, 6*sizeof(float),cudaMemcpyDeviceToHost);
  //  DL = 3.8f;
  DL = 2.9f;
  Xdiv = (int)((Minx[1]-Minx[0])/DL+1);
  Ydiv = (int)((Minx[3]-Minx[2])/DL+1);
  Zdiv = (int)((Minx[5]-Minx[4])/DL+1);
  makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_bounding_xyz, Minx[0], Minx[2], Minx[4],
										 attraction_range, Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL);

  globalrank = 0;

  // open trajectory file
  trajfile = fopen ("traj.xyz", "w");
  if ( trajfile == NULL)
	{
	  printf("Failed to open traj.xyz\n");
	  return -1;
	}
  
  write_traj(1, trajfile); 

  for ( step = 1; step <= Time_steps+1; ++step )
	{
	  //     if ( step < Division_step ) PSS=80.0*Temperature;
	  if ( step < 8000 ) PSS=80.0*Temperature;
	  Pressure = PSS;
  
	  if ( (step/1000)*1000 == step )
		{
		  printf("   time %-8d %d C180s\n",step,No_of_C180s);
		}

	  noofblocks      = No_of_C180s;
	  if ( prevnoofblocks < noofblocks )
        {
		  prevnoofblocks  = noofblocks;
		  //        printf("             no of thread blocks = %d, threadsperblock = %d, no of threads = %ld\n",
		  //             noofblocks, threadsperblock, ((long) noofblocks)*((long) threadsperblock));
        }


	  propagate<<<noofblocks,threadsperblock>>>( No_of_C180s, d_C180_nn, d_C180_sign, 
												 d_XP, d_YP, d_ZP, d_X,  d_Y,  d_Z, d_XM, d_YM, d_ZM,
												 d_CMx, d_CMy, d_CMz,
												 R0, Pressure, Youngs_mod ,
												 internal_damping, delta_t, d_bounding_xyz,
												 attraction_strength, attraction_range,
												 repulsion_strength, repulsion_range,
												 viscotic_damping, mass,
												 Minx[0], Minx[2], Minx[4], Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL);

	  if ( step%Division_step == 0 )
        {
		  printf("   time %-8d", step);

		  CenterOfMass<<<No_of_C180s,256>>>( No_of_C180s,
		  									 d_XP, d_YP, d_ZP, d_CMx, d_CMy, d_CMz);

		  volumes<<<No_of_C180s,192>>>(No_of_C180s, d_C180_56,
									   d_XP, d_YP, d_ZP, d_CMx , d_CMy, d_CMz, d_volume);

		  cudaMemcpy(volume, d_volume, No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);

		  if (countOnlyInternal != 0){
			newCellInds = (int *)calloc(No_of_C180s, sizeof(int)); // The system can only double at maximum
		  }


		  /* No need to print all of this
		  printf(" cell volumes: ");
		  for ( i = 0; i < No_of_C180s; ++i )
            {
			  printf("%f ",  volume[i]); 
			  if ( ((i+1)/5)*5 == i+1 ) printf("\n");
            }
		  printf("\n");
		  */

		  
           
		  temp_No_of_C180s =  No_of_C180s;
		  newcells = 0;
		  for ( rank = 0; rank < temp_No_of_C180s; ++rank ) 
            {
			  globalrank = (globalrank+1)%temp_No_of_C180s;
			  if ( volume[globalrank] < 2.9f )
				continue;
			  //printf("       in cell division, cell %3d divides, new cell #%3d\n",  					 globalrank, No_of_C180s+1);
			  do 
				{
				  ranmar(ran2,2);
				  ran2[0] = 2.0f*ran2[0]-1.0f; 
				  ran2[1] = 2.0f*ran2[1]-1.0f; 
				  s = ran2[0]*ran2[0] + ran2[1]*ran2[1];
				}
			  while ( s >= 1.0f);
			  theta = 3.141592654f/2.0f- acosf(1.0f-2.0f*s);
			  if ( fabsf(ran2[0]) < 0.000001 ) phi = 0.0f;
			  else phi = acos(ran2[0]/sqrtf(s));
			  if ( ran2[1] < 0 ) phi = -phi;
			  //            printf("THETA = %f, PHI = %f\n", theta*180.0f/3.141592654f, phi*180.0f/3.141592654f);
			  ran2[0] = theta; ran2[1] = phi;
                    
			  cudaMemcpy( d_ran2, ran2, 2*sizeof(float),cudaMemcpyHostToDevice);
			  NDIV[rank] += 1;
			  cell_division<<<1,256>>>(globalrank, d_XP, d_YP, d_ZP,  d_X, d_Y, d_Z,  
									   No_of_C180s, d_ran2, repulsion_range);
			  ++No_of_C180s;                
  			  ++newcells;
			  if (countOnlyInternal != 0){
				newCellInds[newcells - 1] = No_of_C180s; // index of the new cell
			  }
			  
			  if ( No_of_C180s > MaxNoofC180s ){printf("Too meny cells: %d\nExiting!\n", No_of_C180s);return(-1);}
			  //            if ( newcells > (int)(0.1f*temp_No_of_C180s) ) break;
            }

		  // Get the cells that are within the radius of gyration

		  if (countOnlyInternal != 0){

			cudaMemcpy(CMx, d_CMx, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(CMy, d_CMy, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(CMz, d_CMz, No_of_C180s*sizeof(float), cudaMemcpyDeviceToHost);

			// Calculate the center of mass of the system

			CoMx = 0;
			CoMy = 0;
			CoMz = 0; 

			for (int cell_ind = 0; cell_ind < No_of_C180s ; cell_ind++) {
			  CoMx += 192 * mass * CMx[cell_ind];
			  CoMy += 192 * mass * CMy[cell_ind];
			  CoMz += 192 * mass * CMz[cell_ind]; 
			}
	  	  
			CoMx = (1/(192*mass*No_of_C180s)) * CoMx;
			CoMy = (1/(192*mass*No_of_C180s)) * CoMy;
			CoMz = (1/(192*mass*No_of_C180s)) * CoMz;

			// calculate the max radius
		  
			Rmax = 0;
			tempX = 0;
			tempY = 0;
			tempZ = 0;

			for (int cell_ind = 0; cell_ind < No_of_C180s; cell_ind++) {
			  tempX = CMx[cell_ind] - CoMx;
			  tempY = CMy[cell_ind] - CoMy;
			  tempZ = CMz[cell_ind] - CoMz;
			  Rmax = max(sqrtf( tempX*tempX + tempY*tempY + tempZ*tempZ ), Rmax);
			}

			if (newcells > 0){
			  newcells_imp = 0; 
			  float dist = Rmax; 
			  for (int new_cell_ind = 0; new_cell_ind < newcells; new_cell_ind++) {
				tempX = CMx[newCellInds[new_cell_ind]] - CoMx;
				tempY = CMy[newCellInds[new_cell_ind]] - CoMy;
				tempZ = CMz[newCellInds[new_cell_ind]] - CoMz;

				dist = sqrtf( tempX*tempX + tempY*tempY + tempZ*tempZ );

				if (dist < 0.6*Rmax){
				  newcells_imp++; 
				}
						  
			  }

			  dist = Rmax;
			  No_of_C180s_imp = 0; 
			  for (int cell_ind = 0; cell_ind < No_of_C180s; cell_ind++) {
				tempX = CMx[cell_ind] - CoMx;
				tempY = CMy[cell_ind] - CoMy;
				tempZ = CMz[cell_ind] - CoMz;
				dist = sqrtf( tempX*tempX + tempY*tempY + tempZ*tempZ );
				if (dist < 0.6*Rmax){
				  No_of_C180s_imp++;
				}
			  }
			}

			dividingCells[(step-1)/Division_step] = newcells_imp;
			totalCells[(step-1)/Division_step] = No_of_C180s_imp;
			printf (" %d new cells born!\n", newcells);
			free(newCellInds); 
		  }
		  else{
		  		  
			dividingCells[(step-1)/Division_step] = newcells;
			totalCells[(step-1)/Division_step] = No_of_C180s;
			printf (" %d new cells born!\n", newcells);
		  }
        }

	  bounding_boxes<<<No_of_C180s,32>>>(No_of_C180s,
										 d_XP,d_YP,d_ZP,d_X,d_Y,d_Z,d_XM,d_YM,d_ZM, 
										 d_bounding_xyz, d_CMx, d_CMy, d_CMz);

	  reductionblocks = (No_of_C180s-1)/1024+1;
	  minmaxpre<<<reductionblocks,1024>>>( No_of_C180s, d_bounding_xyz, 
										   d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
	  minmaxpost<<<1,1024>>>( reductionblocks, d_Minx, d_Maxx, d_Miny, d_Maxy, d_Minz, d_Maxz);
	  cudaMemset(d_NoofNNlist, 0, 1024*1024);

	  cudaMemcpy(Minx, d_Minx, 6*sizeof(float), cudaMemcpyDeviceToHost);
	  Xdiv = (int)((Minx[1]-Minx[0])/DL+1);
	  Ydiv = (int)((Minx[3]-Minx[2])/DL+1);
	  Zdiv = (int)((Minx[5]-Minx[4])/DL+1);
	  makeNNlist<<<No_of_C180s/512+1,512>>>( No_of_C180s, d_bounding_xyz, Minx[0], Minx[2], Minx[4], 
											 attraction_range, Xdiv, Ydiv, Zdiv, d_NoofNNlist, d_NNlist, DL);

 
	  if ( step%trajWriteInt == 0 ) 
		{
		  //printf("   Writing trajectory to traj.xyz...\n"); 
		  cudaMemcpy(X, d_X, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
		  cudaMemcpy(Y, d_Y, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
		  cudaMemcpy(Z, d_Z, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
		  write_traj(step, trajfile); 
		}

	  

	  Temperature += delta_t;
	  myError = cudaGetLastError();
	  if ( cudaSuccess != myError )
		{ printf( "4 Error %d: %s!\n",myError,cudaGetErrorString(myError) );return(-1);}
	}

  // Write postscript file
  cudaMemcpy(X, d_X, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(Y, d_Y, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(Z, d_Z, 192*No_of_C180s*sizeof(float),cudaMemcpyDeviceToHost);
  PSNET(No_of_C180s*270,Side_length,L1,X,Y,Z,CCI);

  printf("Xdiv = %d, Ydiv = %d, Zdiv = %d\n", Xdiv, Ydiv, Zdiv );

  FILE* MitIndFile; 
  MitIndFile = fopen("mit-index.dat", "w");
 
  if (MitIndFile == NULL)
	{
	  printf("Failed to open mit-index.dat\n");
	  exit(1); 
	}

  
  for (int i = 0; i < (Time_steps/Division_step) + 1; i++)
	{
	  if ( dividingCells[i]!=0 && totalCells[i]!=0 )
		fprintf(MitIndFile, "%f\n", (float)dividingCells[i]/totalCells[i]);
	}
   														    
  cudaFree( (void *)d_bounding_xyz );
  cudaFree( (void *)d_XP ); 
  cudaFree( (void *)d_YP );
  cudaFree( (void *)d_ZP );  
  cudaFree( (void *)d_X  );
  cudaFree( (void *)d_Y  );
  cudaFree( (void *)d_Z  );
  cudaFree( (void *)d_XM );
  cudaFree( (void *)d_YM );
  cudaFree( (void *)d_ZM );
  cudaFree( (void *)d_CMx );
  cudaFree( (void *)d_CMy );
  cudaFree( (void *)d_CMz );
  cudaFree( (void *)d_ran2 );

  cudaFree( (void *)d_C180_nn);
  cudaFree( (void *)d_C180_sign);
  free(CMx); free(CMy); free(CMz);
  free(dividingCells); free(totalCells);

  fclose(trajfile); 
  fclose(MitIndFile); 
  return(0);
}



int initialize_C180s(int Orig_No_of_C180s) 
{
  int rank;
  int atom;
  float initx[181], inity[181], initz[181];
  FILE *infil;

  printf("      Initializing positions for %d fullerenes...\n", Orig_No_of_C180s);

  X = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  Y = (float *)calloc(192*MaxNoofC180s,sizeof(float));
  Z = (float *)calloc(192*MaxNoofC180s,sizeof(float));

  bounding_xyz = (float *)calloc(MaxNoofC180s,6*sizeof(float));

  CPUMemory += 3L*192L*MaxNoofC180s*sizeof(float);
  CPUMemory += MaxNoofC180s*6L*sizeof(float);

  infil = fopen("C180","r");
  if ( infil == NULL ) {printf("Unable to open file C180\n");return(-1);}
  for ( atom = 0 ; atom < 180 ; ++atom)
	{
      if ( fscanf(infil,"%f %f %f",&initx[atom], &inity[atom], &initz[atom]) != 3 ) 
		{
		  printf("   Unable to read file C180 on line %d\n",atom+1);
		  fclose(infil);
		  return(-1);
		} 
	}
  fclose(infil);

  ranmar(ran2,Orig_No_of_C180s);
  
  for ( rank = 0; rank < Orig_No_of_C180s; ++rank )
	{
      ey=rank%Side_length;
      ex=rank/Side_length;
      for ( atom = 0 ; atom < 180 ; ++atom)
		{
          X[rank*192+atom] = initx[atom] + L1*ex + 0.5*L1;
          Y[rank*192+atom] = inity[atom] + L1*ey + 0.5*L1;
          Z[rank*192+atom] = initz[atom] + (ran2[rank]-0.5);
		}
	}

  return(0);
}


int generate_random(int no_of_ran1_vectors)    
{
  // This function uses marsaglia random number generator
  // Defined in marsaglia.h
  int seed_ij, seed_kl ,ij,kl;

  ran2 = (float *)calloc(MaxNoofC180s+1,sizeof(float));
  CPUMemory += (MaxNoofC180s+1L)*sizeof(float);

  time_t current_time; 
  time(&current_time);
  seed_ij = (int)current_time;
  localtime(&current_time);
  seed_kl = (int)current_time; 
  ij = seed_ij%31328; 
  kl = seed_kl%30081;
  rmarin(ij,kl);
  
  return(0);
}



int read_fullerene_nn(void) 
{
  int i,end;
  int N1, N2, N3, N4, N5, N6, Sign;
  FILE *infil;

  printf("   Reading C180NN ..\n");

  infil = fopen("C180NN","r");
  if ( infil == NULL ) {printf("Unable to open file C180NN\n");return(-1);}

  end = 180;
  for ( i = 0; i < 180 ; ++i )
	{
	  if ( fscanf(infil,"%d,%d,%d,%d", &N1, &N2, &N3, &Sign) != 4 ) {end = i; break;}
	  C180_nn[0 + i] = N1-1;
	  C180_nn[192+i] = N2-1;
	  C180_nn[384+i] = N3-1;
	  C180_sign[i] = Sign;
	}
  fclose(infil);

  if ( end < 180 ) {printf("Error: Unable to read line %d in file C180NN\n",end);return(-1);}

  printf("   Reading C180C ..\n");

  infil = fopen("C180C","r");
  if ( infil == NULL ) {printf("Unable to open file C180C\n");return(-1);}
  
  end = 270;
  for ( i = 0; i < 270 ; ++i )
	{
	  if ( fscanf(infil,"%d,%d", &N1, &N2) != 2 ) {end = i; break;}
	  CCI[0][i] = N1-1;
	  CCI[1][i] = N2-1;
	}
  fclose(infil);
  
  if ( end < 270 ) {printf("Error: Unable to read line %d in file C180C\n",end);return(-1);}
  
  printf("      read nearest neighbour ids for atoms in C180\n");

  printf("   Reading C180 pentagons, hexagons ..\n");

  infil = fopen("C180_pentahexa","r");
  if ( infil == NULL ) {printf("Unable to open file C180_pentahexa\n");return(-1);}

  end = 12;
  for ( i = 0; i < 12 ; ++i )
	{
      if ( fscanf(infil,"%d %d %d %d %d", &N1, &N2, &N3, &N4, &N5) != 5 ) {end = i; break;}
      C180_56[i*7+0] = N1;
      C180_56[i*7+1] = N2;
      C180_56[i*7+2] = N3;
      C180_56[i*7+3] = N4;
      C180_56[i*7+4] = N5;
      C180_56[i*7+5] = N1;
      C180_56[i*7+6] = N1;
	}
  if ( end != 12 ) {printf("Error: Unable to read line %d in file C180_pentahexa\n",end);return(-1);}
  end = 80;
  for ( i = 0; i < 80 ; ++i )
	{
      if ( fscanf(infil,"%d %d %d %d %d %d", &N1, &N2, &N3, &N4, &N5, &N6) != 6 ) {end = i; break;}
      C180_56[84+i*7+0] = N1;
      C180_56[84+i*7+1] = N2;
      C180_56[84+i*7+2] = N3;
      C180_56[84+i*7+3] = N4;
      C180_56[84+i*7+4] = N5;
      C180_56[84+i*7+5] = N6;
      C180_56[84+i*7+6] = N1;
	}
  if ( end != 80 ) {printf("Error: Unable to read line %d in file C180_pentahexa\n",end);return(-1);}

  fclose(infil);

  return(0);
}

int read_global_params(void)
{
  int error;
  FILE *infil;

  printf("   Reading inp.dat ..\n");

  infil = fopen("inp.dat","r");
  if ( infil == NULL ) {printf("Error: Unable to open file inp.dat\n");return(-1);}
  
  error = 0;

  if ( fscanf(infil,"%f",&mass)                != 1 ) {error = 1;}
  if ( fscanf(infil,"%f",&repulsion_range)     != 1 ) {error = 2;}
  if ( fscanf(infil,"%f",&attraction_range)    != 1 ) {error = 3;}
  if ( fscanf(infil,"%f",&repulsion_strength)  != 1 ) {error = 4;}
  if ( fscanf(infil,"%f",&attraction_strength) != 1 ) {error = 5;}
  if ( fscanf(infil,"%f",&Youngs_mod)          != 1 ) {error = 6;}
  if ( fscanf(infil,"%f",&viscotic_damping)    != 1 ) {error = 7;}
  if ( fscanf(infil,"%f",&internal_damping)    != 1 ) {error = 8;}
  if ( fscanf(infil,"%d",&Division_step)       != 1 ) {error = 9;}
  if ( fscanf(infil,"%d",&Time_steps)          != 1 ) {error = 10;}
  if ( fscanf(infil,"%f",&delta_t)             != 1 ) {error = 11;}
  if ( fscanf(infil,"%d",&Restart)             != 1 ) {error = 12;}
  if ( fscanf(infil,"%d",&trajWriteInt)        != 1 ) {error = 13;}
  if ( fscanf(infil,"%d",&countOnlyInternal)   != 1 ) {error = 14;}
  if ( fscanf(infil,"%f",&radFrac)             != 1 ) {error = 15;}
  fclose(infil);

  if ( error != 0 ) 
	{
	  printf("   Error reading line %d from file inp.dat\n",error);
	  return(-1);
	}

  if ( radFrac < 0.4 || radFrac > 0.8 || radFrac < 0 ){
	countOnlyInternal = 0;
  }

  printf("      mass                = %f\n",mass);
  printf("      spring equilibrium  = %f\n",R0);
  printf("      repulsion range     = %f\n",repulsion_range);
  printf("      attraction range    = %f\n",attraction_range);
  printf("      repulsion strength  = %f\n",repulsion_strength);
  printf("      attraction strength = %f\n",attraction_strength);
  printf("      Youngs modulus      = %f\n",Youngs_mod);
  printf("      viscotic damping    = %f\n",viscotic_damping);
  printf("      internal damping    = %f\n",internal_damping);
  printf("      Division step       = %d\n",Division_step);
  printf("      Time steps          = %d\n",Time_steps);
  printf("      delta t             = %f\n",delta_t);
  printf("      Restart             = %d\n",Restart);
  printf("      trajWriteInterval   = %d\n",trajWriteInt);
  printf("      countOnlyInternal   = %d\n", countOnlyInternal);
  printf("      radFrac             = %f\n", radFrac);

  
  return(0);
}




//C *****************************************************************



__global__ void propagate( int No_of_C180s, int d_C180_nn[], int d_C180_sign[], 
						   float d_XP[], float d_YP[], float d_ZP[], 
						   float d_X[],  float d_Y[],  float d_Z[], 
						   float d_XM[], float d_YM[], float d_ZM[], 
						   float *d_CMx, float *d_CMy, float *d_CMz,
						   float R0, float Pressure, float Youngs_mod ,
						   float internal_damping, float delta_t,
						   float d_bounding_xyz[], 
						   float attraction_strength, float attraction_range,
						   float repulsion_strength, float repulsion_range,
						   float viscotic_damping, float mass,
						   float Minx, float Miny,  float Minz, int Xdiv, int Ydiv, int Zdiv, 
						   int *d_NoofNNlist, int *d_NNlist, float DL)
{
  int rank, atom, nn_rank, nn_atom;
  int N1, N2, N3; 
  int NooflocalNN;
  int localNNs[8];
  float deltaX, deltaY, deltaZ;
  float A1, A2, A3;
  float B1, B2, B3;
  float TX, TY, TZ;
  float NORM, R;
  float NX, NY, NZ;

  rank = blockIdx.x;
  atom = threadIdx.x;


  if ( rank < No_of_C180s && atom < 180 )
	{
	  N1 = d_C180_nn[  0+atom];
	  N2 = d_C180_nn[192+atom];
	  N3 = d_C180_nn[384+atom];
	  A1 = d_X[rank*192+N2]-d_X[rank*192+N1];
	  A2 = d_Y[rank*192+N2]-d_Y[rank*192+N1];
	  A3 = d_Z[rank*192+N2]-d_Z[rank*192+N1];
	  B1 = d_X[rank*192+N3]-d_X[rank*192+N1];
	  B2 = d_Y[rank*192+N3]-d_Y[rank*192+N1];
	  B3 = d_Z[rank*192+N3]-d_Z[rank*192+N1];
	  TX = A2*B3-A3*B2;
	  TY = A3*B1-A1*B3;
	  TZ = A1*B2-A2*B1;
	  NORM = sqrt(TX*TX+TY*TY+TZ*TZ);
	  NX = d_C180_sign[atom]*TX/NORM;
	  NY = d_C180_sign[atom]*TY/NORM;
	  NZ = d_C180_sign[atom]*TZ/NORM;


	  float FX = 0.0f;
	  float FY = 0.0f;
	  float FZ = 0.0f;

	  float X = d_X[rank*192+atom];
	  float Y = d_Y[rank*192+atom];
	  float Z = d_Z[rank*192+atom];
      
	  //  Spring Force calculation within cell
	  //  go through three nearest neighbors
	  for ( int i = 0; i < 3 ; ++i ) 
		{
		  N1 = d_C180_nn[i*192+atom];
		  deltaX = d_X[rank*192+N1]-d_X[rank*192+atom];
		  deltaY = d_Y[rank*192+N1]-d_Y[rank*192+atom];
		  deltaZ = d_Z[rank*192+N1]-d_Z[rank*192+atom];
		  R  = sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ);

		  FX += +Youngs_mod*(R-R0)/R0*deltaX/R+Pressure*NX;
		  FY += +Youngs_mod*(R-R0)/R0*deltaY/R+Pressure*NY;
		  FZ += +Youngs_mod*(R-R0)/R0*deltaZ/R+Pressure*NZ;

		  FX += -(internal_damping/delta_t)*(-deltaX-(d_XM[rank*192+atom]-d_XM[rank*192+N1]));
		  FY += -(internal_damping/delta_t)*(-deltaY-(d_YM[rank*192+atom]-d_YM[rank*192+N1]));
		  FZ += -(internal_damping/delta_t)*(-deltaZ-(d_ZM[rank*192+atom]-d_ZM[rank*192+N1]));
		}


	  NooflocalNN = 0;

	  int startx = (int)((X -Minx)/DL);
	  if ( startx < 0 ) startx = 0;
	  if ( startx >= Xdiv ) startx = Xdiv-1;

	  int starty = (int)((Y - Miny)/DL);
	  if ( starty < 0 ) starty = 0;
	  if ( starty >= Ydiv ) starty = Ydiv-1;

	  int startz = (int)((Z - Minz)/DL);
	  if ( startz < 0 ) startz = 0;
	  if ( startz >= Zdiv ) startz = Zdiv-1;

	  int index = startz*Xdiv*Ydiv + starty*Xdiv + startx;

	  // interfullerene attraction and repulsion
	  for ( int nn_rank1 = 1 ; nn_rank1 <= d_NoofNNlist[index] ; ++nn_rank1 )
		{
		  nn_rank = d_NNlist[32*index+nn_rank1-1];
		  if ( nn_rank == rank ) continue;
           
		  deltaX  = (X-d_bounding_xyz[nn_rank*6+1]>0.0f)*(X-d_bounding_xyz[nn_rank*6+1]);
		  deltaX += (d_bounding_xyz[nn_rank*6+0]-X>0.0f)*(d_bounding_xyz[nn_rank*6+0]-X);

		  deltaY  = (Y-d_bounding_xyz[nn_rank*6+3]>0.0f)*(Y-d_bounding_xyz[nn_rank*6+3]);
		  deltaY += (d_bounding_xyz[nn_rank*6+2]-Y>0.0f)*(d_bounding_xyz[nn_rank*6+2]-Y);

		  deltaZ  = (Z-d_bounding_xyz[nn_rank*6+5]>0.0f)*(Z-d_bounding_xyz[nn_rank*6+5]);
		  deltaZ += (d_bounding_xyz[nn_rank*6+4]-Z>0.0f)*(d_bounding_xyz[nn_rank*6+4]-Z);
 
		  if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > attraction_range*attraction_range ) continue;
		  ++NooflocalNN;
		  if ( NooflocalNN > 8 ) {printf("Recoverable error: NooflocalNN = %d, should be < 8\n",NooflocalNN);continue;}
		  localNNs[NooflocalNN-1] = nn_rank;
		}
          
	  for ( int i = 0; i < NooflocalNN; ++i )
		{
		  nn_rank =localNNs[i];

		  for ( nn_atom = 0; nn_atom < 180 ; ++nn_atom )
			{

			  deltaX = d_X[rank*192+atom]-d_X[nn_rank*192+nn_atom];
			  deltaY = d_Y[rank*192+atom]-d_Y[nn_rank*192+nn_atom];
			  deltaZ = d_Z[rank*192+atom]-d_Z[nn_rank*192+nn_atom];

			  R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;
			  if ( R >= attraction_range*attraction_range ) continue;
			  R = sqrt(R);

			  if ( R < attraction_range ) 
				{
				  FX += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
				  FY += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
				  FZ += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;
				}
			  if ( R < repulsion_range ) 
				{
				  FX += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX;
				  FY += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY;
				  FZ += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;
				  if ( deltaX*(d_CMx[rank]-d_CMx[nn_rank])  +
					   deltaY*(d_CMy[rank]-d_CMy[nn_rank])  +
					   deltaZ*(d_CMz[rank]-d_CMz[nn_rank]) < 0.0f )
					printf("fullerene %d inside %d?\n",rank, nn_rank);
				}

			}

		}



	  // time propagation

	  d_XP[rank*192+atom] = 
		1.0/(1.0+viscotic_damping*delta_t/(2*mass))*
		((delta_t*delta_t/mass)*FX+2*d_X[rank*192+atom]+(viscotic_damping*delta_t/(2*mass)-1.0)*d_XM[rank*192+atom]);
	  d_YP[rank*192+atom] = 
		1.0/(1.0+viscotic_damping*delta_t/(2*mass))*
		((delta_t*delta_t/mass)*FY+2*d_Y[rank*192+atom]+(viscotic_damping*delta_t/(2*mass)-1.0)*d_YM[rank*192+atom]);
	  d_ZP[rank*192+atom] = 
		1.0/(1.0+viscotic_damping*delta_t/(2*mass))*
		((delta_t*delta_t/mass)*FZ+2*d_Z[rank*192+atom]+(viscotic_damping*delta_t/(2*mass)-1.0)*d_ZM[rank*192+atom]);

     
	}




}




void write_traj(int t_step, FILE* trajfile)
{

  fprintf(trajfile, "%d\n", No_of_C180s * 192);
  fprintf(trajfile, "Step: %d\n", t_step); 

  for (int p = 0; p < No_of_C180s*192; p++)
	{
	  fprintf(trajfile, "C  %.7f  %.7f  %.7f\n", X[p], Y[p], Z[p]);
	}
}
