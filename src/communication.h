#ifndef STD_LIB
#include <stdio.h>
#include <stdlib.h>
#endif


#include "mpi.h"
#include <cuda.h>


#define CudaErrorCheck() { \
        cudaError_t e = cudaPeekAtLastError();        \
        if (e!=cudaSuccess){                                            \
            printf("Cuda failure in %s, line %d, code %d %s\n", __FILE__, __LINE__, e, cudaGetErrorString(e)); \
            exit(0); \
        }\
    }
    
    

void Send_Recv_ghost_cells( int No_of_Ghost_cells_buffer, int No_of_Ghost_cells, int receiver, int sender, int tag, MPI_Comm cart_comm,
			     int shift_sender, int shift_receiver,	
			     float* X_gc_buffer, float* Y_gc_buffer, float* Z_gc_buffer, float* velListX_gc_buffer, float* velListY_gc_buffer, float* velListZ_gc_buffer,
			     float* CMx_gc_buffer, float* CMy_gc_buffer, float* CMz_gc_buffer,
			     float* X_gc, float* Y_gc, float* Z_gc, float* velListX_gc, float* velListY_gc, float* velListZ_gc, float* CMx_gc,
			     float* CMy_gc, float* CMz_gc)
{

			
			//printf("I am here, ghost\n");
			
    			MPI_Sendrecv( X_gc_buffer + shift_sender*192, 192*No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			X_gc + shift_receiver*192, No_of_Ghost_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( Y_gc_buffer + shift_sender*192, 192*No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			Y_gc + shift_receiver*192, No_of_Ghost_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 			
                 	MPI_Sendrecv( Z_gc_buffer + shift_sender*192, 192*No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			Z_gc + shift_receiver*192, No_of_Ghost_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);

    	
    	    		MPI_Sendrecv( velListX_gc_buffer + shift_sender*192, 192*No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			velListX_gc + shift_receiver*192, No_of_Ghost_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( velListY_gc_buffer + shift_sender*192, 192*No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			velListY_gc + shift_receiver*192, No_of_Ghost_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 			
                 	MPI_Sendrecv( velListZ_gc_buffer + shift_sender*192, 192*No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			velListZ_gc + shift_receiver*192, No_of_Ghost_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);

    		
    		    	MPI_Sendrecv( CMx_gc_buffer + shift_sender, No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			CMx_gc + shift_receiver, No_of_Ghost_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( CMy_gc_buffer + shift_sender, No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			CMy_gc + shift_receiver, No_of_Ghost_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 			
                 	MPI_Sendrecv( CMz_gc_buffer + shift_sender, No_of_Ghost_cells_buffer, MPI_FLOAT, receiver, tag,
                 			CMz_gc + shift_receiver, No_of_Ghost_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
    

}



void Send_Recv_migrated_cells(int No_of_migrated_cells_buffer, int No_of_migrated_cells, int receiver, int sender, int tag, MPI_Comm cart_comm, 
			     int shift_sender,int shift_receiver,
			     float* X_mc_buffer, float* Y_mc_buffer, float* Z_mc_buffer, float* velListX_mc_buffer, float* velListY_mc_buffer, float* velListZ_mc_buffer,
			     float* CMx_mc_buffer, float* CMy_mc_buffer, float* CMz_mc_buffer, float* ScaleFactor_mc_buffer, float* Youngs_mod_mc_buffer, float* Growth_rate_mc_buffer,
			     float* DivisionVolume_mc_buffer, float* gamma_env_mc_buffer, float* viscotic_damp_mc_buffer, float* pressList_mc_buffer,float* Apo_rate_mc_buffer,
			     float* squeeze_rate_mc_buffer, int* CellINdex_mc_buffer,	
			     float* X_mc, float* Y_mc, float* Z_mc, float* velListX_mc,
			     float* velListY_mc, float* velListZ_mc,float* CMx_mc, float* CMy_mc, float* CMz_mc, float* ScaleFactor_mc, float* Youngs_mod_mc, float* Growth_rate_mc,
			     float* DivisionVolume_mc, float* gamma_env_mc, float* viscotic_damp_mc, float* pressList_mc,float* Apo_rate_mc, float* squeeze_rate_mc, int* CellINdex_mc,
			     bool colloidal_dynamics)
{
		    	
		    	
		    	//printf("I am here, migaration\n");
		    	
		    	MPI_Sendrecv( X_mc_buffer + shift_sender*192, 192*No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			X_mc + 192*shift_receiver  , No_of_migrated_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( Y_mc_buffer + shift_sender*192, 192*No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			Y_mc + 192*shift_receiver, No_of_migrated_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
		    	
		    	MPI_Sendrecv( Z_mc_buffer + shift_sender*192, 192*No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			Z_mc + 192*shift_receiver, No_of_migrated_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
			
			
		    	
		    	MPI_Sendrecv( velListX_mc_buffer + shift_sender*192, 192*No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			velListX_mc + 192*shift_receiver, No_of_migrated_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( velListY_mc_buffer + shift_sender*192, 192*No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			velListY_mc + 192*shift_receiver, No_of_migrated_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
		    	
		    	MPI_Sendrecv( velListZ_mc_buffer + shift_sender*192, 192*No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			velListZ_mc + 192*shift_receiver, No_of_migrated_cells*192, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 			
		    	
		    	MPI_Sendrecv( CMx_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			CMx_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
		    	
		    	MPI_Sendrecv( CMy_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			CMy_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
		    	
		    	MPI_Sendrecv( CMz_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			CMz_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
			
			MPI_Sendrecv( gamma_env_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			gamma_env_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
		    	
		    	MPI_Sendrecv( viscotic_damp_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			viscotic_damp_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
			
		    	MPI_Sendrecv( ScaleFactor_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			ScaleFactor_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( Youngs_mod_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			Youngs_mod_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);

                 	MPI_Sendrecv( pressList_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 			pressList_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE); 
                 			
                 	MPI_Sendrecv( CellINdex_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_INT, receiver, tag,
                 			CellINdex_mc + shift_receiver, No_of_migrated_cells, MPI_INT, sender, tag, cart_comm, MPI_STATUS_IGNORE);                	
                 	
                 	if(!colloidal_dynamics) {
                 	
                 		MPI_Sendrecv( Growth_rate_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 				Growth_rate_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 		MPI_Sendrecv( DivisionVolume_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 				DivisionVolume_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 		MPI_Sendrecv( Apo_rate_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 				Apo_rate_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 		MPI_Sendrecv( squeeze_rate_mc_buffer + shift_sender, No_of_migrated_cells_buffer, MPI_FLOAT, receiver, tag,
                 				squeeze_rate_mc + shift_receiver, No_of_migrated_cells, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
			}
                 			
}



void Send_Recv_ghost_ECM( int No_of_Ghost_ECM_buffer, int No_of_Ghost_ECM, int receiver, int sender, int tag, MPI_Comm cart_comm,
			     int shift_sender, int shift_receiver,	
			     float* ECM_x_buffer, float* ECM_y_buffer, float* ECM_z_buffer,
			     float* ECM_x_ecm, float* ECM_y_ecm, float* ECM_z_ecm,
			     bool ind_comm, int* ECM_ind_buffer, int* ECM_ind_ecm)
{

    		
    		    	MPI_Sendrecv( ECM_x_buffer + shift_sender, No_of_Ghost_ECM_buffer, MPI_FLOAT, receiver, tag,
                 			ECM_x_ecm + shift_receiver, No_of_Ghost_ECM, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 	
                 	MPI_Sendrecv( ECM_y_buffer + shift_sender, No_of_Ghost_ECM_buffer, MPI_FLOAT, receiver, tag,
                 			ECM_y_ecm + shift_receiver, No_of_Ghost_ECM, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
                 			
                 	MPI_Sendrecv( ECM_z_buffer + shift_sender, No_of_Ghost_ECM_buffer, MPI_FLOAT, receiver, tag,
                 			ECM_z_ecm + shift_receiver, No_of_Ghost_ECM, MPI_FLOAT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
    

			if (ind_comm) MPI_Sendrecv( ECM_ind_buffer + shift_sender, No_of_Ghost_ECM_buffer, MPI_INT, receiver, tag,
                 					ECM_ind_ecm + shift_receiver, No_of_Ghost_ECM, MPI_INT, sender, tag, cart_comm, MPI_STATUS_IGNORE);
}

	
