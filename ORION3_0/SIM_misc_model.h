/*-------------------------------------------------------------------------
 *                             ORION 2.0 
 *
 *         					Copyright 2009 
 *  	Princeton University, and Regents of the University of California 
 *                         All Rights Reserved
 *
 *                         
 *  ORION 2.0 was developed by Bin Li at Princeton University and Kambiz Samadi at
 *  University of California, San Diego. ORION 2.0 was built on top of ORION 1.0. 
 *  ORION 1.0 was developed by Hangsheng Wang, Xinping Zhu and Xuning Chen at 
 *  Princeton University.
 *
 *  If your use of this software contributes to a published paper, we
 *  request that you cite our paper that appears on our website 
 *  http://www.princeton.edu/~peh/orion.html
 *
 *  Permission to use, copy, and modify this software and its documentation is
 *  granted only under the following terms and conditions.  Both the
 *  above copyright notice and this permission notice must appear in all copies
 *  of the software, derivative works or modified versions, and any portions
 *  thereof, and both notices must appear in supporting documentation.
 *
 *  This software may be distributed (but not offered for sale or transferred
 *  for compensation) to third parties, provided such third parties agree to
 *  abide by the terms and conditions of this notice.
 *
 *  This software is distributed in the hope that it will be useful to the
 *  community, but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 *
 *-----------------------------------------------------------------------*/

#ifndef _SIM_MISC_MODEL_H
#define _SIM_MISC_MODEL_H

typedef enum {
	RESULT_BUS = 1,
	GENERIC_BUS,
	BUS_MAX_MODEL
} SIM_power_bus_model_t;

typedef enum {
	GENERIC_SEL = 1,
	SEL_MAX_MODEL
} SIM_power_sel_model_t;

typedef enum {
	NEG_DFF = 1,	/* negative egde-triggered D flip-flop */
	FF_MAX_MODEL
} SIM_power_ff_model_t;

typedef enum {
	IDENT_ENC = 1,	/* identity encoding */
	TRANS_ENC,	/* transition encoding */
	BUSINV_ENC,	/* bus inversion encoding */
	BUS_MAX_ENC
} SIM_power_bus_enc_t;


#endif	/* _SIM_MISC_MODEL_H */
