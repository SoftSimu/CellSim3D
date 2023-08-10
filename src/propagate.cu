#include<cuda.h>
#include<vector_functions.h>
#include<stdio.h>
#include<curand_kernel.h>
#include "VectorFunctions.hpp"


//__constant__ float d_dt;

__global__ void DeviceRandInit(curandState *rngStates, uint *d_seeds, unsigned long long num){
    size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < num){
        curandState rS = rngStates[idx];
        curand_init(d_seeds[idx], 0, 0 , &rS);
        rngStates[idx] = rS; 
    }
}



#define check_float3(a){ \
        if (!good_float3(a)){ \
        printf("uh oh hotdog\n"); \
        printf("%d, %s, %d, %d \n", __LINE__, __FILE__, blockIdx.x, threadIdx.x);   \
        asm("trap;"); \
    } \
}

__device__ float3 GetAngleForce(const float3 iPos, const float3 kPos,
                                      const float theta_o, const float k){
    float i_dot_k = dot(iPos, kPos);
    float ri_2    = mag2(iPos);
    float rk_2    = mag2(kPos);
    //if (ri_2*rk_2 - i_dot_k*i_dot_k < 0) asm("trap;");
    float c1 = -1.0f/( sqrtf( ri_2*rk_2 - i_dot_k*i_dot_k + 1.0e-3f));

    float c2 = i_dot_k/ri_2;

    float angle = i_dot_k/(sqrtf(ri_2)*sqrtf(rk_2));
    if ( angle < -0.995f ) angle = -0.995f;                   //JW
    if ( angle > +0.995f ) angle = +0.995f;                   //JW

    float theta = acos(angle);
//    float theta = acos(i_dot_k/(sqrtf(ri_2)*sqrtf(rk_2) + 1e-3));

    float3 F_i = -k * c1 * (theta - theta_o)*(kPos - c2*iPos);

    // float imag = mag(iPos);
    // float kmag = mag(kPos);
    
    // float cos_theta = dot(iPos, kPos)/(imag*kmag);
    // float cos_thetao = cos(theta_o); 

    // float3 F_i =  -2*k/(imag*kmag) * (cos_theta - cos_thetao) * kPos; 
    
    if (!good_float3(F_i)){
        printf("c1: %f, c2: %f, theta: %f, %d %d\n", c1, c2, theta, (int)blockIdx.x, (int)threadIdx.x);
        printf("i.k %f ri2 %f rk2 %f, %d %d\n", i_dot_k, ri_2, rk_2, (int)blockIdx.x, (int)threadIdx.x);
        printf("iPos %f, %f, %f, and kPos %f, %f, %f\n", iPos.x, iPos.y, iPos.z, kPos.x, kPos.y, kPos.z);
        asm("trap;");
    }

    // if (!good_float3(F_i)){
    //     printf("Angle force calculation failed for node %d in cell %d\n", blockIdx.x, threadIdx.x);
    //     asm("trap;");
    // }
    
    return F_i; 
}


__device__ float3 GetAngleForceECM(const float3 iPos, const float3 kPos,
                                      const float theta_o, const float k){
    float i_dot_k = dot(iPos, kPos);
    float ri_2    = mag2(iPos);
    float rk_2    = mag2(kPos);
    
    float c1 = -1.0f/( sqrtf( ri_2*rk_2 - i_dot_k*i_dot_k + 1.0e-3f));
    float c2 = i_dot_k/ri_2;

    float angle = i_dot_k/(sqrtf(ri_2)*sqrtf(rk_2));
    if ( angle < -0.995f ) angle = -0.995f;                   
    if ( angle > +0.995f ) angle = +0.995f; 
    
    float theta = acos(angle);

    float3 F_i = -k * c1 * (theta - theta_o)*(kPos - c2*iPos);
    
    if (!good_float3(F_i)){
        printf("c1: %f, c2: %f, theta: %f, theta_o: %f, %d %d\n", c1, c2, theta, theta_o, blockIdx.x, threadIdx.x);
        printf("i.k %f ri2 %f rk2 %f, %d %d\n", i_dot_k, ri_2, rk_2, blockIdx.x, threadIdx.x);
        printf("iPos %f, %f, %f, and kPos %f, %f, %f\n", iPos.x, iPos.y, iPos.z, kPos.x, kPos.y, kPos.z);
        asm("trap;");
    }
    
    return F_i; 
}



// Watch me whip, whatch me...
__device__ void NeighNeighs (const int nodeInd, const int ni, int& nii, int& nij, int& nik,
                             const angles3 d_theta0[], float& theta1_o, float& theta2_o){
    if (nii == nodeInd){
        theta1_o = d_theta0[ni].aij;
        theta2_o = d_theta0[ni].aik; 
        nii = nij;
        nij = nik;
    } else if (nij == nodeInd){
        theta1_o = d_theta0[ni].aij;
        theta2_o = d_theta0[ni].ajk; 
        nij = nik;
    } else {
        theta1_o = d_theta0[ni].aik;
        theta2_o = d_theta0[ni].ajk; 
    }
}

// This fucking function will break if we decide to make cell geometry more interesting
__device__ float3 CalculateAngleForce(int nodeInd, int d_C180_nn[],
                                      float d_X[], float d_Y[], float d_Z[],
                                      const angles3 d_theta0[], float k, int cellInd){
    // First get the first angle contribution
    int ni = d_C180_nn[0*192 + nodeInd];
    int nj = d_C180_nn[1*192 + nodeInd];
    int nk = d_C180_nn[2*192 + nodeInd];
    
    float3 nodePos, nodeForce;

    nodeForce = make_float3(0, 0, 0);
    nodePos = make_float3(d_X[cellInd*192 + nodeInd], 
                          d_Y[cellInd*192 + nodeInd], 
                          d_Z[cellInd*192 + nodeInd]);

    float3 niPos, njPos, nkPos;
    niPos = make_float3(d_X[cellInd*192 + ni],
                        d_Y[cellInd*192 + ni],
                        d_Z[cellInd*192 + ni]);
    
    njPos = make_float3(d_X[cellInd*192 + nj],
                        d_Y[cellInd*192 + nj],
                        d_Z[cellInd*192 + nj]);

    nkPos = make_float3(d_X[cellInd*192 + nk],
                        d_Y[cellInd*192 + nk],
                        d_Z[cellInd*192 + nk]);
    
    angles3 nodeAngles = d_theta0[nodeInd]; 
    
    nodeForce = nodeForce - 
        (GetAngleForce(niPos-nodePos, njPos-nodePos, nodeAngles.aij, k) + 
         GetAngleForce(njPos-nodePos, niPos-nodePos, nodeAngles.aij, k));
    
    nodeForce = nodeForce - 
        (GetAngleForce(njPos-nodePos, nkPos-nodePos, nodeAngles.ajk, k) + 
         GetAngleForce(nkPos-nodePos, njPos-nodePos, nodeAngles.ajk, k));

    nodeForce = nodeForce -
        (GetAngleForce(niPos-nodePos, nkPos-nodePos, nodeAngles.aik, k) +
         GetAngleForce(nkPos-nodePos, niPos-nodePos, nodeAngles.aik, k));

    
    // Now second angle contributions
    // Each neighbor will have two other neighbors + our node

    int nii = d_C180_nn[ni + 0*192];
    int nij = d_C180_nn[ni + 1*192]; 
    int nik = d_C180_nn[ni + 2*192];

    int nji = d_C180_nn[nj + 0*192];
    int njj = d_C180_nn[nj + 1*192]; 
    int njk = d_C180_nn[nj + 2*192];

    int nki = d_C180_nn[nk + 0*192];
    int nkj = d_C180_nn[nk + 1*192]; 
    int nkk = d_C180_nn[nk + 2*192];
    
    float theta1_o, theta2_o; 
    float3 tiPos, tjPos;
    tiPos = make_float3(0, 0, 0);
    tjPos = make_float3(0, 0, 0);
    
    NeighNeighs(nodeInd, ni, nii, nij, nik, d_theta0, theta1_o, theta2_o);
    tiPos = make_float3(d_X[cellInd*192 + nii],
                        d_Y[cellInd*192 + nii],
                        d_Z[cellInd*192 + nii]);

    tjPos = make_float3(d_X[cellInd*192 + nij],
                        d_Y[cellInd*192 + nij],
                        d_Z[cellInd*192 + nij]);
    
    nodeForce = nodeForce + GetAngleForce(nodePos - niPos, tiPos - niPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - niPos, tjPos - niPos, theta2_o, k);
    
    NeighNeighs(nodeInd, nj, nji, njj, njk, d_theta0, theta1_o, theta2_o);
    tiPos = make_float3(d_X[cellInd*192 + nji],
                        d_Y[cellInd*192 + nji],
                        d_Z[cellInd*192 + nji]);

    tjPos = make_float3(d_X[cellInd*192 + njj],
                        d_Y[cellInd*192 + njj],
                        d_Z[cellInd*192 + njj]);

    nodeForce = nodeForce + GetAngleForce(nodePos - njPos, tiPos - njPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - njPos, tjPos - njPos, theta2_o, k);

    NeighNeighs(nodeInd, nk, nki, nkj, nkk, d_theta0, theta1_o, theta2_o);
    tiPos = make_float3(d_X[cellInd*192 + nki],
                        d_Y[cellInd*192 + nki],
                        d_Z[cellInd*192 + nki]);
                                 
    tjPos = make_float3(d_X[cellInd*192 + nkj],
                        d_Y[cellInd*192 + nkj],
                        d_Z[cellInd*192 + nkj]);

    nodeForce = nodeForce + GetAngleForce(nodePos - nkPos, tiPos - nkPos, theta1_o, k); 
    nodeForce = nodeForce + GetAngleForce(nodePos - nkPos, tjPos - nkPos, theta2_o, k);

    return nodeForce;
}



__global__ void CalculateConForce_ECM( int Num_ECM,
					float* d_ECM_x, float* d_ECM_y, float* d_ECM_z, 
					float* d_Con_ECM_force_x, float* d_Con_ECM_force_y, float* d_Con_ECM_force_z,
					int* d_ECM_neighbor_updated, int* d_Num_Nei_ECM,
					float* d_R0_ECM, float *d_stiffness_ecm,
					float angleConstant_ecm, float* d_theta0_ECM,
					int* d_ECM_neighbor)
{

	
	int NodeInd = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( NodeInd < Num_ECM)
	{
	
	
		float R, R0, nodeAngles, ECM_stiffness;
		int ni, nj;
	
		float3 nodePos, niPos, njPos;
		float3 Fi, Fj;	
		float3 delta;
		float3 nodeForce = make_float3(0.f, 0.f, 0.f);
		
		nodePos = make_float3(d_ECM_x[NodeInd], d_ECM_y[NodeInd], d_ECM_z[NodeInd]);
				
		int num_nei = d_Num_Nei_ECM[NodeInd];

		//printf("Node is %d, Num Nei is %d, old ind %d, new ind %d\n", NodeInd, d_Num_Nei_ECM[NodeInd], d_ECM_neighbor[32*NodeInd], d_ECM_neighbor_updated[32*NodeInd]);
		
		for ( int nn_rank1 = 0; nn_rank1 < num_nei; ++nn_rank1 )
		{
		
			int loc = 32*NodeInd + nn_rank1;
			
			ni = d_ECM_neighbor_updated[loc];
			nj = d_ECM_neighbor_updated[loc+1];
			
			//if(ni == 0) printf("Node is %d, nn_rank is %d, old list %d\n", NodeInd, nn_rank1, d_ECM_neighbor[loc]);
			
			niPos = make_float3( d_ECM_x[ni], d_ECM_y[ni], d_ECM_z[ni]);
			
			delta = niPos - nodePos;

			R = sqrt(delta.x*delta.x+delta.y*delta.y+delta.z*delta.z);
			
			R0 = d_R0_ECM[loc];
			ECM_stiffness = d_stiffness_ecm[loc];
			
			//if(R != R0) printf("R is %f and R0 is %f, node is %d, loc is %d, ni is %d\n",R,R0,NodeInd,loc,ni);
			
			//spring forces
        	    	nodeForce.x += ECM_stiffness*(R-R0)/R0*delta.x/R;
        	    	nodeForce.y += ECM_stiffness*(R-R0)/R0*delta.y/R;
        	    	nodeForce.z += ECM_stiffness*(R-R0)/R0*delta.z/R;


    			//angle forces		
    			njPos = make_float3( d_ECM_x[nj], d_ECM_y[nj], d_ECM_z[nj]);
    			
    			nodeAngles = d_theta0_ECM[loc];
    			//nodeAngles = d_theta0_ECM.ak[loc]; 

			Fi = GetAngleForceECM(niPos-nodePos, njPos-nodePos, nodeAngles, angleConstant_ecm);
			Fj = GetAngleForceECM(njPos-nodePos, niPos-nodePos, nodeAngles, angleConstant_ecm);
			
			nodeForce = nodeForce - (Fi + Fj);
			
			atomicAdd( &d_Con_ECM_force_x[ni] , Fi.x); 
			atomicAdd( &d_Con_ECM_force_y[ni] , Fi.y); 
			atomicAdd( &d_Con_ECM_force_z[ni] , Fi.z);
			
			atomicAdd( &d_Con_ECM_force_x[nj] , Fj.x); 
			atomicAdd( &d_Con_ECM_force_y[nj] , Fj.y);
			atomicAdd( &d_Con_ECM_force_z[nj] , Fj.z); 
		
		
		}

		// need to change this
		atomicAdd( &d_Con_ECM_force_x[NodeInd] , nodeForce.x); 
		atomicAdd( &d_Con_ECM_force_y[NodeInd] , nodeForce.y); 
		atomicAdd( &d_Con_ECM_force_z[NodeInd] , nodeForce.z); 
		
		
		//d_Con_ECM_force_x[NodeInd] += nodeForce.x;
        	//d_Con_ECM_force_y[NodeInd] += nodeForce.y;
        	//d_Con_ECM_force_z[NodeInd] += nodeForce.z;

	}


}



__global__ void CalculateDisForce_ECM( int Num_ECM, float* d_ECM_Vx, float* d_ECM_Vy, float* d_ECM_Vz,
					float *d_Dis_ECM_force_x, float *d_Dis_ECM_force_y, float *d_Dis_ECM_force_z,
					int* d_ECM_neighbor_updated, int* d_Num_Nei_ECM,
					float vis_damp_ecm, float gamma_env_ecm)
{
	
	int NodeInd = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( NodeInd < Num_ECM)
	{
	
	        int num_nei = d_Num_Nei_ECM[NodeInd];
		float3 force = make_float3(0.f, 0.f, 0.f);
		
		float3 nodeVelocity = make_float3(d_ECM_Vx[NodeInd],d_ECM_Vy[NodeInd],d_ECM_Vz[NodeInd]); 
		float3 neighVelocity = make_float3(0.f, 0.f, 0.f); 
		
			
		
		int nn_rank;
		for ( int nn_rank1 = 0; nn_rank1 < num_nei; ++nn_rank1 )
		{
			
			nn_rank = d_ECM_neighbor_updated[32*NodeInd + nn_rank1];
			
			neighVelocity = make_float3(d_ECM_Vx[nn_rank], d_ECM_Vy[nn_rank], d_ECM_Vz[nn_rank]);
			
			force = force - vis_damp_ecm*(nodeVelocity - neighVelocity); 	

		}
		
		force = force - gamma_env_ecm*nodeVelocity;
		
		d_Dis_ECM_force_x[NodeInd] = force.x;
        	d_Dis_ECM_force_y[NodeInd] = force.y;
        	d_Dis_ECM_force_z[NodeInd] = force.z;

	
	}



}

        
__global__ void CalculateConForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                           float d_X[],  float d_Y[],  float d_Z[],
                           float *d_CMx, float *d_CMy, float *d_CMz,
                           float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,                           
                           float* d_R0,float* d_ScaleFactor, float* d_pressList, float* d_stiffness, 
                           float attraction_strength, float attraction_range,
                           float repulsion_strength, float repulsion_range,
                           float* d_viscotic_damp,
                           int Xdiv, int Ydiv, int Zdiv, double3 boxMax, 
                           int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float DL, float* d_gamma_env,
                           float threshDist, 
                           double3 BoxMin, float3 Subdivision_min, float Youngs_mod,  float angleConstant, 
                           bool constrainAngles, const angles3 d_theta0[], R3Nptrs d_forceList, R3Nptrs d_ExtForces,
                           bool impurity, float f_range,
                           bool useRigidSimulationBox, bool useRigidBoxZ, bool useRigidBoxY, bool useRigidBoxX,
                           int MaxNeighList,
                           bool ECM,
                           float *d_Con_ECM_force_x, float *d_Con_ECM_force_y, float* d_Con_ECM_force_z,
                           float* d_ECM_x, float* d_ECM_y, float* d_ECM_z,
                           float attraction_strength_ecm, float attraction_range_ecm,
                           float repulsion_strength_ecm, float repulsion_range_ecm,
                           int *d_NoofNNlist_ECM, int *d_NNlist_ECM, float DL_ecm, int Xdiv_ecm, int Ydiv_ecm,
                           int MaxNeighList_ecm)
{


#ifdef FORCE_DEBUG
        __shared__ float FX_sum;
        __shared__ float FY_sum;
        __shared__ float FZ_sum;
        
        if (threadIdx.x == 0){
        
            FX_sum = 0;
            FY_sum = 0;
            FZ_sum = 0;

        
        }

        __syncthreads();

#endif


    	int rank = blockIdx.x;
    	int atom = threadIdx.x;
    	int atomInd = rank*192+atom;    	
    	
    	
    	if ( rank < No_of_C180s && atom < 180 )
    	{

    		float deltaX, deltaY, deltaZ;
    		float R;
    		int N1;
    		int nn_rank;
    		float R0=0;
    		float range;
    		float Pressure = d_pressList[rank];
    		float stiffness = d_stiffness[rank];
    		float Scale = d_ScaleFactor[rank];
    			
    	
        	if (isnan(d_X[atomInd]) ||
        	    isnan(d_Y[atomInd]) || 
        	    isnan(d_Z[atomInd])){
        	    printf("OH SHIT: we have a nan\n");
        	    printf("Particle index: %d, Cell: %d\n", atom, rank);
        	    printf("Crash now :(\n"); 
        	    asm("trap;"); 
        	}


        	float X = d_X[atomInd];
        	float Y = d_Y[atomInd];
        	float Z = d_Z[atomInd];

        	float FX = 0.f;
        	float FY = 0.f;
        	float FZ = 0.f;
        	float FX_ext = 0.f;
        	float FY_ext = 0.f;
        	float FZ_ext = 0.f;

        	int nnAtomInd;


        	for ( int i = 0; i < 3 ; ++i ) // Better to open this loop
        	{
            		N1 = d_C180_nn[i*192+atom];

            		deltaX = d_X[rank*192+N1]-X;
            		deltaY = d_Y[rank*192+N1]-Y;
            		deltaZ = d_Z[rank*192+N1]-Z;

            		R  = sqrt(deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ);

            		R0 = Scale*d_R0[i*192 + atom];

            		//spring forces
            		FX += +stiffness*(R-R0)/R0*deltaX/R;
            		FY += +stiffness*(R-R0)/R0*deltaY/R;
            		FZ += +stiffness*(R-R0)/R0*deltaZ/R;
        	}

        	// new growth force

        	float3 r_CM = make_float3(X - d_CMx[rank], 
               	                  Y - d_CMy[rank], 
               	                  Z - d_CMz[rank]);
        	float3 gForce  = make_float3(0.f, 0.f, 0.f);

        	gForce = 3*Pressure*calcUnitVec(r_CM);

        
        	FX += gForce.x; 
        	FY += gForce.y; 
        	FZ += gForce.z; 

        	if (constrainAngles){
            		
            		float3 t = CalculateAngleForce(atom, d_C180_nn,
                       	                    	d_X, d_Y, d_Z,
                       	                    	d_theta0, angleConstant /*Youngs_mod*/, rank);
            		FX += t.x; FY += t.y; FZ += t.z;
        	}
        
        
        
#ifdef FORCE_DEBUG

        	atomicAdd(&FX_sum, FX);
        	__syncthreads();
        	atomicAdd(&FY_sum, FY);
        	__syncthreads();
        	atomicAdd(&FZ_sum, FZ);
        	__syncthreads();
        	if (threadIdx.x == 0){
            		printf("Spring, pressure, internal\n");
            		printf("Fx = %f, Fy = %f, Fz = %f\n", FX_sum, FY_sum, FZ_sum);
        	}

#endif

        	// interfullerene attraction and repulsion
        

        	int posX = (int)((X - Subdivision_min.x)/DL);
        	if ( posX < 0 ) posX = 0;
        	if ( posX > Xdiv-1 ) posX = Xdiv-1;


		int posY = (int)((Y - Subdivision_min.y)/DL);
        	if ( posY < 0 ) posY = 0;
        	if ( posY > Ydiv-1 ) posY = Ydiv-1;
         
         
        	int posZ = (int)((Z - Subdivision_min.z)/DL);
        	if ( posZ < 0 ) posZ = 0;
        	if ( posZ > Zdiv-1 ) posZ = Zdiv-1;
        

        	int index = posZ*Xdiv*Ydiv + posY*Xdiv + posX;


	        float3 contactForce = make_float3(0.f, 0.f, 0.f);
        
	        for ( int nn_rank1 = 0; nn_rank1 < d_NoofNNlist[index]; ++nn_rank1 )
	        {
	            
	            	nn_rank = d_NNlist[MaxNeighList*index + nn_rank1];
	            
	            	if ( nn_rank == rank )
	                	continue;
                
	            	deltaX  = X - d_CMx[nn_rank];
	            	deltaY  = Y - d_CMy[nn_rank];                
	            	deltaZ  = Z - d_CMz[nn_rank];
	            
		    	//range = f_range*d_ScaleFactor[nn_rank] + attraction_range;
		    	range = f_range + attraction_range;	
                    
	            	if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ >  range*range)
	            	    continue;
		    
		    	
		    	nnAtomInd = nn_rank*192;
		    	for (int nn_atom = 0; nn_atom < 180; ++nn_atom )
            	    	{
               		 
               		 //nnAtomInd += nn_atom;

               		 deltaX = X - d_X[nnAtomInd+nn_atom];
               		 deltaY = Y - d_Y[nnAtomInd+nn_atom];
               		 deltaZ = Z - d_Z[nnAtomInd+nn_atom];
        
            
               		 R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;

               		 R = sqrt(R);
               		 
               		 if ( R >= attraction_range )
               		     continue;

               		 contactForce.x += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
               		 contactForce.y += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
               		 contactForce.z += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;


               		 if ( R <= repulsion_range )
               		 {
               		     //if (R < (repulsion_range-0.01)) R = repulsion_range-0.01; 
               		     contactForce.x += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX;
               		     contactForce.y += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY;
               		     contactForce.z += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;
               		 }
            	    
            	    	}
	        
	        
	        }      	
        	
        	if (impurity){
        	
        		
        		for ( int nn_rank1 = 0; nn_rank1 < d_NoofNNlistPin[index] ; ++nn_rank1 )
	        	{
	        
	            		nn_rank = d_NNlistPin[32*index+nn_rank1];
                
	            		deltaX  = X - d_CMxPin[nn_rank];
	            		deltaY  = Y - d_CMyPin[nn_rank];                
	            		deltaZ  = Z - d_CMzPin[nn_rank];

				range = f_range + attraction_range;
               
	            		if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > range*range )
	                		continue;


        	    		nnAtomInd = nn_rank*192;
            			for (int nn_atom = 0; nn_atom < 180 ; ++nn_atom )
            			{
               			
               			//nnAtomInd += nn_atom;
	
               			deltaX = X - d_XPin[nnAtomInd+nn_atom];
               			deltaY = Y - d_YPin[nnAtomInd+nn_atom];
               			deltaZ = Z - d_ZPin[nnAtomInd+nn_atom];
        
            
               		 	R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;

               		 	if ( R >= attraction_range*attraction_range )
               		     		continue;

               		 	R = sqrt(R);


               		     	contactForce.x += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaX;
               		     	contactForce.y += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaY;
               		     	contactForce.z += -attraction_strength*Youngs_mod*(attraction_range-R)/R*deltaZ;

               		 	if ( R <= repulsion_range )
               		 	{
               		     	 
               		     		contactForce.x += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaX;
               		     		contactForce.y += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaY;
               		     		contactForce.z += +repulsion_strength*Youngs_mod*(repulsion_range-R)/R*deltaZ;
               		 	}

           		 	}

        		}
        	
        	}
        	
		if (ECM) {
			
			
			float3 F_ecm = make_float3(0.f, 0.f, 0.f);
			
			
			posX = (int)((X - Subdivision_min.x)/DL_ecm);
        		if ( posX < 0 ) posX = 0;
        		if ( posX > Xdiv_ecm - 1 ) posX = Xdiv_ecm-1;


			posY = (int)((Y - Subdivision_min.y)/DL_ecm);
        		if ( posY < 0 ) posY = 0;
        		if ( posY > Ydiv_ecm - 1 ) posY = Ydiv_ecm -1;
         
        		index = posY*Xdiv_ecm + posX;

		
			for ( int nn_rank1 = 0; nn_rank1 < d_NoofNNlist_ECM[index] ; ++nn_rank1 )
	        	{
				
				nn_rank = d_NNlist_ECM[MaxNeighList_ecm*index+nn_rank1];
				
				//printf("nn_rank is %d\n",nn_rank);
				
				deltaX  =  X - d_ECM_x[nn_rank];
	            		deltaY  =  Y - d_ECM_y[nn_rank];                
	            		deltaZ  =  Z - d_ECM_z[nn_rank];
	            		
	            		R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;

               		R = sqrt(R);
               		
               		if ( R >= attraction_range_ecm )
               		     		continue;
				 
				F_ecm.x = -attraction_strength_ecm*Youngs_mod*(attraction_range_ecm-R)/R*deltaX;
				F_ecm.y = -attraction_strength_ecm*Youngs_mod*(attraction_range_ecm-R)/R*deltaY;
				F_ecm.z = -attraction_strength_ecm*Youngs_mod*(attraction_range_ecm-R)/R*deltaZ;
				 
				 
               		if ( R <= repulsion_range_ecm )
               		{
               		     F_ecm.x += +repulsion_strength_ecm*Youngs_mod*(repulsion_range_ecm-R)/R*deltaX;
               		     F_ecm.y += +repulsion_strength_ecm*Youngs_mod*(repulsion_range_ecm-R)/R*deltaY;
               		     F_ecm.z += +repulsion_strength_ecm*Youngs_mod*(repulsion_range_ecm-R)/R*deltaZ;
               		}
				 
				
				atomicAdd( &d_Con_ECM_force_x[nn_rank] , -F_ecm.x); 
				atomicAdd( &d_Con_ECM_force_y[nn_rank] , -F_ecm.y);
				atomicAdd( &d_Con_ECM_force_z[nn_rank] , -F_ecm.z); 
			
				contactForce = contactForce + F_ecm;
			
			}
		
		
		}


		FX += contactForce.x;
        	FY += contactForce.y;
        	FZ += contactForce.z; 
        	
        	FX_ext += contactForce.x;
        	FY_ext += contactForce.y;
        	FZ_ext += contactForce.z; 

#ifdef FORCE_DEBUG

        	if (threadIdx.x == 0){
            		FX_sum = 0;
            		FY_sum = 0;
            		FZ_sum = 0;
        	}
        	__syncthreads();

        	atomicAdd(&FX_sum, FX);
        	__syncthreads();
        	atomicAdd(&FY_sum, FY);
        	__syncthreads();
        	atomicAdd(&FZ_sum, FZ);
        	__syncthreads();
        	if (threadIdx.x == 0){
            		printf("neighbours\n");
            		printf("Fx = %f, Fy = %f, Fz = %f\n", FX_sum, FY_sum, FZ_sum);
        	}

#endif
        	// add forces from simulation box if needed:
        	
        	if (useRigidSimulationBox){
        	
        		float gap1, gap2; 

        		gap1 = X - BoxMin.x;
       		gap2 = boxMax.x - X; 

        		if (gap1 < threshDist){
            			FX += -100*Youngs_mod*(gap1 - threshDist);
            			FX_ext += -100*Youngs_mod*(gap1 - threshDist);
        		}

        		if (gap2 < threshDist){
            			FX += 100*Youngs_mod*(gap2 - threshDist);
            			FX_ext += 100*Youngs_mod*(gap2 - threshDist);
        		}
            
        		gap1 = Y - BoxMin.y;
        		gap2 = boxMax.y - Y;

        		if (gap1 < threshDist){
            			FY += -100*Youngs_mod*(gap1 - threshDist);
            			FY_ext += -100*Youngs_mod*(gap1 - threshDist);
        		}

        		if (gap2 < threshDist){
            			FY += 100*Youngs_mod*(gap2 - threshDist);
            			FY_ext += 100*Youngs_mod*(gap2 - threshDist);
        		}

        		gap1 = Z - BoxMin.z;
        		gap2 = boxMax.z - Z;
	
        		if (gap1 < threshDist){
            			FZ += -100*Youngs_mod*(gap1 - threshDist);
            			FZ_ext += -100*Youngs_mod*(gap1 - threshDist);
        		}
	
        		if (gap2 < threshDist){
            			FZ += 100*Youngs_mod*(gap2 - threshDist);
            			FZ_ext += 100*Youngs_mod*(gap2 - threshDist);
        		}
	
		} else {
		
		        if (useRigidBoxZ){
            
            			float gap1, gap2; 

            			gap1 = Z;
            			gap2 = boxMax.z - Z;

            			if (gap1 < threshDist){
               			FZ += -100*Youngs_mod*(gap1 - threshDist);
                			FZ_ext += -100*Youngs_mod*(gap1 - threshDist);
            			}

            			if (gap2 < threshDist){
                			FZ += 100*Youngs_mod*(gap2 - threshDist);
                			FZ_ext += 100*Youngs_mod*(gap2 - threshDist);
            			}

        		}
        
        		if (useRigidBoxY){
        
            			float gap1, gap2; 

            			gap1 = Y;
            			gap2 = boxMax.y - Y;

            			if (gap1 < threshDist){
                			FY += -100*Youngs_mod*(gap1 - threshDist);
               			FY_ext += -100*Youngs_mod*(gap1 - threshDist);
                
            			}

            			if (gap2 < threshDist){
                			FY += 100*Youngs_mod*(gap2 - threshDist);
                			FY_ext += 100*Youngs_mod*(gap2 - threshDist);
            			}

        		}
        
        		if (useRigidBoxX){
        
            			float gap1, gap2; 

            			gap1 = X;
            			gap2 = boxMax.x - X;

            			if (gap1 < threshDist){
                			FX += -100*Youngs_mod*(gap1 - threshDist);
                			FX_ext += -100*Youngs_mod*(gap1 - threshDist);
                
            			}

            			if (gap2 < threshDist){
                			FX += 100*Youngs_mod*(gap2 - threshDist);
                			FX_ext += 100*Youngs_mod*(gap2 - threshDist);
            			}

        	       }
		
		}
        

        	d_forceList.x[atomInd] = FX;
        	d_forceList.y[atomInd] = FY;
        	d_forceList.z[atomInd] = FZ;
       	
       	d_ExtForces.x[atomInd] = contactForce.x;
        	d_ExtForces.y[atomInd] = contactForce.y;
       	d_ExtForces.z[atomInd] = contactForce.z;
   	
       	//d_ExtForces.x[atomInd] = FX_ext;
        	//d_ExtForces.y[atomInd] = FY_ext;
       	//d_ExtForces.z[atomInd] = FZ_ext;

   	
   	
   	}

}


__global__ void CalculateDisForce( int No_of_C180s, int d_C180_nn[], int d_C180_sign[],
                                   float d_X[],  float d_Y[],  float d_Z[],
                                   float *d_CMx, float *d_CMy, float *d_CMz,
                           	    float d_XPin[],  float d_YPin[],  float d_ZPin[],
                           	    float *d_CMxPin, float *d_CMyPin, float *d_CMzPin,                                   
                                   float gamma_int,
                                   float attraction_range,
                                   float* d_viscotic_damp, float* d_ScaleFactor,
                                   int Xdiv, int Ydiv, int Zdiv, float3 Subdivision_min,
                                   int *d_NoofNNlist, int *d_NNlist, int *d_NoofNNlistPin, int *d_NNlistPin, float DL, float* d_gamma_env,
                                   float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_fDisList, R3Nptrs d_ConFricForces, bool impurity, float f_range,
                                   int MaxNeighList,
                                   bool ECM,
                                   float *d_Dis_ECM_force_x, float *d_Dis_ECM_force_y, float *d_Dis_ECM_force_z,
                                   float* d_ECM_x, float* d_ECM_y, float* d_ECM_z,
				    float* d_ECM_Vx, float* d_ECM_Vy, float* d_ECM_Vz,
				    float attraction_range_ecm, float vis_ecm_cell,
                           	    int *d_NoofNNlist_ECM, int *d_NNlist_ECM, float DL_ecm, int Xdiv_ecm, int Ydiv_ecm,
                           	    int MaxNeighList_ecm)
{
    
    
    size_t cellInd = blockIdx.x;
    size_t nodeInd = threadIdx.x;
    


    	if (cellInd < No_of_C180s && nodeInd < 180){


    	
        	size_t globalNodeInd = cellInd*192 + nodeInd;
        	size_t N = 0;
		float range;
        	
        	float3 force = make_float3(0, 0, 0);
        	float3 force_fric = make_float3(0, 0, 0);
        	
        	float3 nodeVelocity = make_float3(d_velListX[globalNodeInd],
                                          	   d_velListY[globalNodeInd],
                                          	   d_velListZ[globalNodeInd]);

        	float3 neighVelocity = make_float3(0, 0, 0);
        
        	// dampen bonding
        	for (int i = 0; i < 3; ++i){
        	
        	    N = d_C180_nn[i*192+nodeInd];
        	    neighVelocity = make_float3(d_velListX[cellInd*192+N],
        	                                d_velListY[cellInd*192+N],
        	                                d_velListZ[cellInd*192+N]);
                
        	    force = force - gamma_int*(nodeVelocity - neighVelocity);
        	}

        	// inter-cellular friction
        	// This break if cell geometry every changes
        	int N1 = d_C180_nn[  0+nodeInd];
        	int N2 = d_C180_nn[192+nodeInd];
        	int N3 = d_C180_nn[384+nodeInd];

        	float3 A = make_float3(d_X[cellInd*192+N2]-d_X[cellInd*192+N1],
        	                       d_Y[cellInd*192+N2]-d_Y[cellInd*192+N1],
        	                       d_Z[cellInd*192+N2]-d_Z[cellInd*192+N1]);

        	float3 B = make_float3(d_X[cellInd*192+N3]-d_X[cellInd*192+N1],
        	                       d_Y[cellInd*192+N3]-d_Y[cellInd*192+N1],
        	                       d_Z[cellInd*192+N3]-d_Z[cellInd*192+N1]);

        	float3 normal = calcUnitVec(cross(A, B));

        	normal = d_C180_sign[nodeInd]*normal;

        
        	float X = d_X[globalNodeInd];
        	float Y = d_Y[globalNodeInd];
        	float Z = d_Z[globalNodeInd];
        

        	float deltaX = 0;
        	float deltaY = 0;
        	float deltaZ = 0;
        	float R = 0;
        	int nn_rank = 0;
        	int nnAtomInd = 0;

        	int posX = 0;    
        	int posY = 0;
        	int posZ = 0;

		
        	posX = (int)((X - Subdivision_min.x)/DL);
        	if ( posX < 0 ) posX = 0;
        	if ( posX > Xdiv-1 ) posX = Xdiv-1;
       
        	posY = (int)((Y - Subdivision_min.y)/DL);
        	if ( posY < 0 ) posY = 0;
        	if ( posY > Ydiv-1 ) posY = Ydiv-1;
        
        	posZ = (int)((Z - Subdivision_min.z)/DL);
        	if ( posZ < 0 ) posZ = 0;
        	if ( posZ > Zdiv-1 ) posZ = Zdiv-1;
        
        
        	int index = posZ*Xdiv*Ydiv + posY*Xdiv + posX;
        
        	for ( int nn_rank1 = 1 ; nn_rank1 < d_NoofNNlist[index] ; ++nn_rank1 )
        	{
        	    	
        	    	nn_rank = d_NNlist[MaxNeighList*index+nn_rank1-1]; 
        	    	
        	    	if ( nn_rank == cellInd ) continue;

        	    	deltaX  = X - d_CMx[nn_rank];
        	    	deltaY  = Y - d_CMy[nn_rank];            
        	    	deltaZ  = Z - d_CMz[nn_rank];

		    	//range = f_range*d_ScaleFactor[nn_rank] + attraction_range;
		    	range = f_range + attraction_range;
		    	
        		if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > range*range )
        	        	continue;
        		
        		
			nnAtomInd = nn_rank*192;
        	    	for ( int nn_atom = 0; nn_atom < 180 ; ++nn_atom )
        	    	{
                
                		//nnAtomInd += nn_atom;
                		
                		deltaX = X - d_X[nnAtomInd+nn_atom];   
        	        	deltaY = Y - d_Y[nnAtomInd+nn_atom];
        	        	deltaZ = Z - d_Z[nnAtomInd+nn_atom];                
                

        	        	R = deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ;
	
	
        	        	if ( R > attraction_range*attraction_range )
        	        	    continue;
	
        	        	neighVelocity = make_float3(d_velListX[nnAtomInd+nn_atom],
        	        	                            d_velListY[nnAtomInd+nn_atom],
        	        	                            d_velListZ[nnAtomInd+nn_atom]);

	
				float3 v_ij = nodeVelocity - neighVelocity;
		
        	        	// Tangential component of relative velocity
        	        	float3 vTau = v_ij - dot(v_ij, normal)*normal;
        	        	force = force - d_viscotic_damp[cellInd]*vTau;
        	        	
				force_fric = force_fric - d_viscotic_damp[cellInd]*vTau;

        	        	
        	    	}

        	
        	}
        	
        	if (impurity){
	
			
			for ( int nn_rank1 = 1 ; nn_rank1 < d_NoofNNlistPin[index] ; ++nn_rank1 )
        		{
        	    		
        	    		nn_rank = d_NNlistPin[32*index+nn_rank1-1]; 

        	    		deltaX  = X - d_CMxPin[nn_rank];
        	    		deltaY  = Y - d_CMyPin[nn_rank];            
        	    		deltaZ  = Z - d_CMzPin[nn_rank];

				range = f_range + attraction_range;
				
        	    		if ( deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ > range*range )
        	        		continue;

				nnAtomInd = nn_rank*192;
	        	    	for ( int nn_atom = 0; nn_atom < 180 ; ++nn_atom )
	        	    	{
                			
                			//nnAtomInd += nn_atom;
                			
	        	        	deltaX = X - d_XPin[nnAtomInd+nn_atom];   
	        	        	deltaY = Y - d_YPin[nnAtomInd+nn_atom];
	        	        	deltaZ = Z - d_ZPin[nnAtomInd+nn_atom];                
                

	        	        	R = deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ;
	
	
	        	        	if ( R > attraction_range*attraction_range )
	        	            		continue;
		
		
	        	        	// Tangential component of the node velocity
	        	        	float3 vTau = nodeVelocity - dot(nodeVelocity, normal)*normal;
	        	        	force = force - d_viscotic_damp[cellInd]*vTau;
	        	    	}

	        	}  	
        	

        	}
        	
        	
        	if (ECM){
			
			float3 F_ecm = make_float3(0, 0, 0);
			
			posX = (int)((X - Subdivision_min.x)/DL_ecm);
        		if ( posX < 0 ) posX = 0;
        		if ( posX > Xdiv_ecm - 1 ) posX = Xdiv_ecm-1;


			posY = (int)((Y - Subdivision_min.y)/DL_ecm);
        		if ( posY < 0 ) posY = 0;
        		if ( posY > Ydiv_ecm - 1 ) posY = Ydiv_ecm -1;
         
        		index = posY*Xdiv_ecm + posX;
			
			for ( int nn_rank1 = 0; nn_rank1 < d_NoofNNlist_ECM[index] ; ++nn_rank1 )
	        	{
				
				nn_rank = d_NNlist_ECM[MaxNeighList_ecm*index+nn_rank1];
				
				deltaX  = X - d_ECM_x[nn_rank];
	            		deltaY  = Y - d_ECM_y[nn_rank];                
	            		deltaZ  = Z - d_ECM_z[nn_rank];
	            		
	            		R = deltaX*deltaX+deltaY*deltaY+deltaZ*deltaZ;

               		R = sqrt(R);
               		
               		if ( R >= attraction_range_ecm )
               		     		continue;

				
				neighVelocity = make_float3(d_ECM_Vx[nn_rank],d_ECM_Vy[nn_rank],d_ECM_Vz[nn_rank]);

	
				float3 v_ij = nodeVelocity - neighVelocity;
				//float3 vTau = v_ij - dot(v_ij, normal)*normal;
				
        	    		// Tangential component of the node velocity
        	    		F_ecm = vis_ecm_cell*v_ij;
        	    		
        	    		atomicAdd( &d_Dis_ECM_force_x[nn_rank] , F_ecm.x); 
				atomicAdd( &d_Dis_ECM_force_y[nn_rank] , F_ecm.y);
				atomicAdd( &d_Dis_ECM_force_z[nn_rank] , F_ecm.z); 	
		
				force = force - F_ecm;
				force_fric = force_fric - F_ecm;
				
			}
				
			
        	
        	}
        	
        	
        	
        	// viscous drag  
        	force = force - d_gamma_env[cellInd]*nodeVelocity;
        
        	// write forces to global memory
        	
        	d_fDisList.x[globalNodeInd] = force.x; 
        	d_fDisList.y[globalNodeInd] = force.y; 
        	d_fDisList.z[globalNodeInd] = force.z; 
        		
        	d_ConFricForces.x[globalNodeInd] = force_fric.x;
        	d_ConFricForces.y[globalNodeInd] = force_fric.y;
        	d_ConFricForces.z[globalNodeInd] = force_fric.z;	
        	
        		
	} 



}


__global__ void CalculateRanForce(int No_of_C180s, curandState *d_rngStates, float rand_scale_factor,
                                  R3Nptrs d_fRanList){
    
    size_t nodeIdx = blockIdx.x*blockDim.x + threadIdx.x;
    //const int cellInd = blockIdx.x;
    
    
    	if (nodeIdx < No_of_C180s*192){
        	curandState rngState = d_rngStates[nodeIdx];
        	d_fRanList.x[nodeIdx] = rand_scale_factor*curand_normal(&rngState); 
        	d_fRanList.y[nodeIdx] = rand_scale_factor*curand_normal(&rngState); 
        	d_fRanList.z[nodeIdx] = rand_scale_factor*curand_normal(&rngState);
        	d_rngStates[nodeIdx] = rngState;
    	}
    
}


__global__ void Integrate(float *d_X, float *d_Y, float *d_Z, 
                          float *d_velListX, float *d_velListY, float *d_velListZ, 
                          float dt, float m,
                          R3Nptrs d_fConList, R3Nptrs d_fDisList, R3Nptrs d_fRanList,
                          int numCells)
{
    
    const int cellInd = blockIdx.x;
    const int node = threadIdx.x;
 
    
    if ( cellInd < numCells && node < 180){
        
        int nodeInd = cellInd*192 + node;
        const float root_dt = sqrtf(dt);
        //if(nodeInd == 0) printf("dt is:	%f",d_dt);    
        d_velListX[nodeInd] = d_velListX[nodeInd] + 0.5*(dt*d_fConList.x[nodeInd] + dt*d_fDisList.x[nodeInd] + \
                                                         root_dt*d_fRanList.x[nodeInd])/m;
        
        d_velListY[nodeInd] = d_velListY[nodeInd] + 0.5*(dt*d_fConList.y[nodeInd] + dt*d_fDisList.y[nodeInd] + \
                                                         root_dt*d_fRanList.y[nodeInd])/m;
        
        d_velListZ[nodeInd] = d_velListZ[nodeInd] + 0.5*(dt*d_fConList.z[nodeInd] + dt*d_fDisList.z[nodeInd] + \
                                                         root_dt*d_fRanList.z[nodeInd])/m;


	d_X[nodeInd] += d_velListX[nodeInd]*dt; 
	d_Y[nodeInd] += d_velListY[nodeInd]*dt; 
	d_Z[nodeInd] += d_velListZ[nodeInd]*dt; 
         

    }

}

__global__ void VelocityUpdateA(float* d_VX, float* d_VY, float* d_VZ,
                                R3Nptrs fConList, R3Nptrs fRanList,
                                float dt, long int num_nodes, float m){
   
    //const int cellInd = blockIdx.x;
    long int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    
    	if (nodeInd < num_nodes){
        	float root_dt = sqrtf(dt);
        	d_VX[nodeInd] = d_VX[nodeInd] + 0.5*(dt*fConList.x[nodeInd] + root_dt*fRanList.x[nodeInd])/m;
        	d_VY[nodeInd] = d_VY[nodeInd] + 0.5*(dt*fConList.y[nodeInd] + root_dt*fRanList.y[nodeInd])/m;
        	d_VZ[nodeInd] = d_VZ[nodeInd] + 0.5*(dt*fConList.z[nodeInd] + root_dt*fRanList.z[nodeInd])/m;
    	}


}


__global__ void VelocityUpdateB(float* d_VX, float* d_VY, float* d_VZ,
                                R3Nptrs fDisList, float dt, long int num_nodes, float m ){
    

    long int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;

    	
    	if (nodeInd < num_nodes){
    	    d_VX[nodeInd] = d_VX[nodeInd] + 0.5*dt*(fDisList.x[nodeInd])/m;
    	    d_VY[nodeInd] = d_VY[nodeInd] + 0.5*dt*(fDisList.y[nodeInd])/m;
    	    d_VZ[nodeInd] = d_VZ[nodeInd] + 0.5*dt*(fDisList.z[nodeInd])/m;
    	}

}


__global__ void Integrate_ECM (float* d_ECM_x, float* d_ECM_y, float* d_ECM_z,
                          float* d_ECM_Vx, float* d_ECM_Vy, float* d_ECM_Vz,
                          float* d_Con_ECM_force_x, float* d_Con_ECM_force_y, float* d_Con_ECM_force_z,
                          float* d_Dis_ECM_force_x, float* d_Dis_ECM_force_y, float *d_Dis_ECM_force_z,
                          float dt, float m_ecm,
                          int Num_ECM, bool Clamped, int* d_clamped_node)
{
 
    int nodeInd = blockIdx.x*blockDim.x + threadIdx.x; 
    
    
    if ( nodeInd < Num_ECM){
    
    
        if (Clamped){
        	

    		if (d_clamped_node[nodeInd] == 0){
    	
    			float factor = (0.5*dt)/m_ecm;
    	
    	
        		d_ECM_Vx[nodeInd] = d_ECM_Vx[nodeInd] + factor*(d_Con_ECM_force_x[nodeInd] + d_Dis_ECM_force_x[nodeInd]);
        		d_ECM_Vy[nodeInd] = d_ECM_Vy[nodeInd] + factor*(d_Con_ECM_force_y[nodeInd] + d_Dis_ECM_force_y[nodeInd]);
        		d_ECM_Vz[nodeInd] = d_ECM_Vz[nodeInd] + factor*(d_Con_ECM_force_z[nodeInd] + d_Dis_ECM_force_z[nodeInd]);
        	
			d_ECM_x[nodeInd] += d_ECM_Vx[nodeInd]*dt; 
			d_ECM_y[nodeInd] += d_ECM_Vy[nodeInd]*dt;
			d_ECM_z[nodeInd] += d_ECM_Vz[nodeInd]*dt;
			
			//printf("I am here");  
        
        	}
         
        } else {
        
            	float factor = (0.5*dt)/m_ecm;
    	
    	
        	d_ECM_Vx[nodeInd] = d_ECM_Vx[nodeInd] + factor*(d_Con_ECM_force_x[nodeInd] + d_Dis_ECM_force_x[nodeInd]);
        	d_ECM_Vy[nodeInd] = d_ECM_Vy[nodeInd] + factor*(d_Con_ECM_force_y[nodeInd] + d_Dis_ECM_force_y[nodeInd]);
        	d_ECM_Vz[nodeInd] = d_ECM_Vz[nodeInd] + factor*(d_Con_ECM_force_z[nodeInd] + d_Dis_ECM_force_z[nodeInd]);
        	
		d_ECM_x[nodeInd] += d_ECM_Vx[nodeInd]*dt; 
		d_ECM_y[nodeInd] += d_ECM_Vy[nodeInd]*dt;
		d_ECM_z[nodeInd] += d_ECM_Vz[nodeInd]*dt; 
		
		 
        
       }

    }

}


__global__ void VelocityUpdateA_ECM(float* d_ECM_Vx, float* d_ECM_Vy, float* d_ECM_Vz,
                                float* d_Con_ECM_force_x, float* d_Con_ECM_force_y, float* d_Con_ECM_force_z,
                                float dt, int Num_ECM, float m_ecm,
                                bool Clamped, int* d_clamped_node)
{
   

    	int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;
    
    	if (nodeInd < Num_ECM){
    		
    		if (Clamped) {
    			
    			if (d_clamped_node[nodeInd] == 0){
        			
        			float factor = (0.5*dt)/m_ecm;
        	
        			d_ECM_Vx[nodeInd] = d_ECM_Vx[nodeInd] + factor*d_Con_ECM_force_x[nodeInd];
        			d_ECM_Vy[nodeInd] = d_ECM_Vy[nodeInd] + factor*d_Con_ECM_force_y[nodeInd];
        			d_ECM_Vz[nodeInd] = d_ECM_Vz[nodeInd] + factor*d_Con_ECM_force_z[nodeInd];
    			}
    		
    		} else {
    		
    		        float factor = (0.5*dt)/m_ecm;
        	
        		d_ECM_Vx[nodeInd] = d_ECM_Vx[nodeInd] + factor*d_Con_ECM_force_x[nodeInd];
        		d_ECM_Vy[nodeInd] = d_ECM_Vy[nodeInd] + factor*d_Con_ECM_force_y[nodeInd];
        		d_ECM_Vz[nodeInd] = d_ECM_Vz[nodeInd] + factor*d_Con_ECM_force_z[nodeInd];
    		
    		
    		}
       }
}


__global__ void VelocityUpdateB_ECM(float* d_ECM_Vx, float* d_ECM_Vy, float* d_ECM_Vz,
                                float* d_Dis_ECM_force_x, float* d_Dis_ECM_force_y, float *d_Dis_ECM_force_z,
                                float dt, int Num_ECM, float m_ecm,
                                bool Clamped, int* d_clamped_node)
{
    

    	int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;
	
    	if (nodeInd < Num_ECM){
    	
    		
    		if (Clamped){
    	    		
    	    		if (d_clamped_node[nodeInd] == 0){
    	    			
    	    			float factor = (0.5*dt)/m_ecm;
    	    	
    	    			d_ECM_Vx[nodeInd] = d_ECM_Vx[nodeInd] + factor*d_Dis_ECM_force_x[nodeInd];
    	    			d_ECM_Vy[nodeInd] = d_ECM_Vy[nodeInd] + factor*d_Dis_ECM_force_y[nodeInd];
    	    			d_ECM_Vz[nodeInd] = d_ECM_Vz[nodeInd] + factor*d_Dis_ECM_force_z[nodeInd];
	   		}
	   		
	   	} else {
	   
	      		float factor = (0.5*dt)/m_ecm;
    	    	
    	    		d_ECM_Vx[nodeInd] = d_ECM_Vx[nodeInd] + factor*d_Dis_ECM_force_x[nodeInd];
    	    		d_ECM_Vy[nodeInd] = d_ECM_Vy[nodeInd] + factor*d_Dis_ECM_force_y[nodeInd];
    	    		d_ECM_Vz[nodeInd] = d_ECM_Vz[nodeInd] + factor*d_Dis_ECM_force_z[nodeInd];
    	    		
	   	}	
    
    	}

}


__global__ void CorrectCoMVelocity_ecm(int Num_ECM, float* d_ECM_Vx, float* d_ECM_Vy, float* d_ECM_Vz, R3Nptrs d_sysCM_All)
{
    
    long int node = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (node < Num_ECM){

        d_ECM_Vx[node] -= *d_sysCM_All.x;       
        d_ECM_Vy[node] -= *d_sysCM_All.y;
        d_ECM_Vz[node] -= *d_sysCM_All.z;
     
    }


}


__global__ void ForwardTime(float *d_XP, float *d_YP, float *d_ZP,
                            float *d_X, float *d_Y, float *d_Z, 
                            int numCells, int impurityNum){
    
    //const int cellInd = blockIdx.x;
    const int nodeInd = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (nodeInd < 192*numCells){
        
        // if (d_XP[nodeInd] != d_XM[nodeInd] ||
        //     d_YP[nodeInd] != d_YM[nodeInd] || 
        //     d_ZP[nodeInd] != d_ZM[nodeInd] ){
        //     printf("%.20f != %.20f or\n%.20f != %.20f or\n%.20f != %.20f\nnodeInd=%d\n",
        //            d_XP[nodeInd], d_XM[nodeInd],
        //            d_YP[nodeInd], d_YM[nodeInd],
        //            d_ZP[nodeInd], d_ZM[nodeInd], nodeInd);
        //     asm("trap;");
        // }
    
        	d_X[nodeInd] = d_XP[nodeInd];
        	d_Y[nodeInd] = d_YP[nodeInd];
        	d_Z[nodeInd] = d_ZP[nodeInd];
    	}
}

__global__ void CorrectCoMMotion( int No_cells_All, float* d_X, float* d_Y, float* d_Z,
                                 R3Nptrs d_sysCM, R3Nptrs d_sysCM_All, double3 BoxCen, long int numParts){
    
    
    __shared__ float Cmx;
    __shared__ float Cmy;
    __shared__ float Cmz;
    
    int tid = threadIdx.x;
    long int partInd = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (tid == 0){
	
	*d_sysCM.x = *d_sysCM_All.x/No_cells_All;
	*d_sysCM.y = *d_sysCM_All.y/No_cells_All;
	*d_sysCM.z = *d_sysCM_All.z/No_cells_All;    
    	
    	Cmx = *d_sysCM.x - BoxCen.x;
    	Cmy = *d_sysCM.y - BoxCen.y;
    	Cmz = *d_sysCM.z - BoxCen.z;
    
    }
    
    __syncthreads();
    

    if (partInd < numParts){
    
        d_X[partInd] -= Cmx;
        d_Y[partInd] -= Cmy;
        d_Z[partInd] -= Cmz;
    
    }

}



__global__ void CorrectCoMVelocity(int No_cells_All, float* d_velListX, float* d_velListY, float* d_velListZ,
                                   R3Nptrs d_sysVCM, R3Nptrs d_sysCM_All, long int numParts){

    __shared__ float VCmx;
    __shared__ float VCmy;
    __shared__ float VCmz;
    
    int tid = threadIdx.x;
    long int partInd = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (tid == 0){
    
    	VCmx = *d_sysCM_All.x;
    	VCmy = *d_sysCM_All.y;
    	VCmz = *d_sysCM_All.z;
    
    }
    
    __syncthreads();


    if (partInd < numParts){

        d_velListX[partInd] -= VCmx;       
        d_velListY[partInd] -= VCmy;
        d_velListZ[partInd] -= VCmz;    
        
    }


}



__global__ void SumForces(R3Nptrs fConList, R3Nptrs fDisList, R3Nptrs fRanList,
                          R3Nptrs fList, 
                          long int numNodes){
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < numNodes){
        fList.x[idx] = fConList.x[idx] + fDisList.x[idx] + fRanList.x[idx]; 
        fList.y[idx] = fConList.y[idx] + fDisList.y[idx] + fRanList.y[idx]; 
        fList.z[idx] = fConList.z[idx] + fDisList.z[idx] + fRanList.z[idx];
    }
}

