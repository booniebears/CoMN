/**
\mainpage
ORION 3.0: A Power and Area Simulator for On-Chip Networks
=================================================================  
Copyright  2012   The Regents of the University of California
All Rights Reserved
 
Permission to copy, modify and distribute any part of this ORION3.0 software distribution for educational, 
research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided 
that the above copyright notice, this paragraph and the following three paragraphs appear in all copies.
 
Those desiring to incorporate this ORION 3.0 software distribution into commercial products or use for 
commercial purposes should contact the Technology Transfer Office.

Technology Transfer Office
University of California, San Diego 
9500 Gilman Drive 
Mail Code 0910 
La Jolla, CA 92093-0910

Ph: (858) 534-5815
FAX: (858) 534-7345
E-MAIL:invent@ucsd.edu.

 
IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, 
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS ORION 3.0 
SOFTWARE DISTRIBUTION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
 
THE ORION 3.0 SOFTWARE DISTRIBUTION PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF 
CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.  
THE UNIVERSITY OF CALIFORNIA MAKES NO REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY KIND, EITHER 
IMPLIED OR EXPRESS, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS 
FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE ORION 3.0 SOFTWARE DISTRIBUTION WILL NOT INFRINGE ANY 
PATENT, TRADEMARK OR OTHER RIGHTS.
 
(1) OVERVIEW  

	ORION 3.0 is a power and area simulator for interconnection networks built on top of ORION 2.0. The structure 
	of ORION 3.0 is different from that of ORION 2.0. ORION 3.0 implements a new class structure as well as new 
	models for router microarchitectural blocks.
	These models are significantly enhanced compared to ORION 2.0, and they are highly accurate with respect to 
	physical implementation.  
	Here is a summary of the major changes: 
	 --	We provide the option to use more accurate models developed through several regression techniques.
	 -- We use a class-based implementation. 
	 --	We provide an option to run multiple parametric as well as non-parametric regression techniques. The user may 
		use our default training data sets or provide their own. 

	Details of ORION 3.0 can be obtained from our DAC12 paper, <TECH REPORT>, and ASPDAC:
	http://vlsicad.ucsd.edu/Publications/Conferences/285/c285.pdf
	<TECH REPORT>
	<ASPDAC>

 
(2) BUILD  

    ORION 3.0 runs under CentOS (5.5 and below). It should also run 
    under other standard Linux distributions with the addition of the correct libraries. 
 
    Before building ORION 3.0, you need to specify router configurations in SIM_port.h file.
    The SIM_port.h file defines the router and link microarchitectural parameters as well as
    technology parameters. Some explanations about how to choose the parameters are provided 
    in Section (4) below.
 
    Procedure to build: just type "make". This will generate one executable file for power and area
	estimation: orion_router 
 
(3) USAGE  
 
    command: orion_router [-v <version>] [-pm] [-d <print_depth>] [-l <load>] <router_name> [-model <model>] [-train <area training file absolute path> <power training file absolute path>] [-test <testfile absolute path>]
 
	Description of options:
	
	-v: version of the tool to be used.
		2 - ORION 2.0.
		3 - ORION 3.0.
		default - ORION 3.0
	
	Options specific to ORION 2.0:
	
	-p: output buffer power, crossbar power, virtual channel allocator power, switch 
		allocator power, clock power and total power. 
    -m: if present, compute maximum power rather than average power. 
    -d depth: output detailed component energy dissipation with varying depth. Try 
		different numbers to see the effects on printout 
    -l load: load is in the range of (0,1), and the default value is 1. Here load 
		corresponds to the probability at which flits arrive at each port. ORION 
		assumes 0.5 switching probability for each arrived flit internally.
	router_name: an arbitrary string as the name of the router.
 	
	New options for ORION 3.0:
	
	-model: type of model to be used for area and power estimation.
	
			basic - ORION 3.0 parametric models
			lsqr  - ORION 3.0 parametric models and LSQR
			rbf   - Radial Basis Function
			kg    - Kriging
			mars  - Multivariate Adaptive Regression
			svm   - Support Vector Machine
			default - basic model, ORION 3.0 parametric models

	-train: use user-supplied training file. User to provide absolute path for training file.
	-test: use user-supplied testing file. User to provide  absolute path for testing file.
	
	
    For ORION 3.0, orion_router outputs power in mW, area in um^2.
    For ORION 2.0, orion_router outputs power in mW, area in um^2, and energy in J. 

(4) FILES 
 
    In SIM_port.h file, we provide various router microarchitectural choices as 
    well as technological choices. Here, we list several parameter choices for 
    your reference. 

(4.1) Technology Parameters 

    (a) PARM_TECH_POINT 
	
	ORION 3.0 supports only 65nm and 45nm technologies.
	
    ORION 2.0 supports 800nm, 400nm, 350nm, 250nm, 180nm, 110nm,
    90nm, 65nm, 45nm and 32nm technologies. ORION 2.0 updates values for 90nm, 
    65nm, 45nm and 32nm while keeping the original values from ORION 1.0 for 800nm to 110nm.
	
    (b) PARM_TRANSISTOR_TYPE  
   	
    PARM_TRANSISTOR_TYPE can be set to HVT, NVT, or LVT. Here HVT means high VT, NVT means 
    normal VT and LVT means low VT. LVT corresponds to high performance router. NVT 
    corresponds to normal operating performance router and HVT corresponds to low 
    power low performance router. 

    (c) PARM_Vdd
	
    PARM_Vdd is the operating voltage in Volt.

    (d) PARM_Freq

    PARM_Freq is the operating frequency in Hz. 

    Note that if your router operates at high frequency, you should use LVT and set
    PARM_Vdd to a higher value. If your router operates at low frequency, you should
    use HVT and set PARM_Vdd to a lower value. If your router operates at normal
    frequency, you should use NVT and set PARM_Vdd to the value in between the LVT and
    HVT's PARM_Vdd values. For example, if you simulate the router at 65nm technology,
    reasonable settings for PARM_Vdd could be 1.2V for LVT and 0.8V for HVT.

(4.2) Router Parameters

    (a) PARM_v_class 
		
    PARM_v_class is the number of message classes in each router. 

    (b) PARM_v_channel 

    PARM_v_channel is the number of virtual channels within each message class.

    (c) PARM_in_buf_set 

    PARM_in_buf_set is the number of input buffers. Note that if you set PARM_in_share_buf 
    to 0, which means that input virtual channels don't physically share buffers, then 
    PARM_in_buf_set is the number of buffers per virtual channel. If you set 
    PARM_in_share_buf to 1, which means that input virtual channels physically share 
    buffers, then PARM_in_buf_set is the total number of buffers that are shared.
	
	(d) PARM_out_buf_set

(4.3) Link Wire Parameters 
    
    (a) WIRE_LAYER_TYPE 
    	
    WIRE_LAYER_TYPE can be set to INTERMEDIATE or GLOBAL. INTERMEDIATE and GLOBAL 
    wiring regimes are used for on chip interconnection.  When selecting these 
    regimes, appropriate wire dimensions will be used to derive the interconnect 
    capacitance that is used in link power calculation. The values to use should 
    be either INTERMEDIATE or GLOBAL depending on the application. Typically, 
    in a 9-layer design, M2-M6 represent INTERMEDIATE, and M7-M9 represent GLOBAL 
    wiring regimes, respectively. 
 

    (b) PARM_width_spacing 
 
    Choices for PARM_width_spacing are SWIDTH_SSPACE, SWIDTH_DSPACE, DWIDTH_SSPACE
    and DWIDTH_DSPACE. "S" and "D" stand for single and double, respectively.
    Width-spacing configuration is a design style which enable designers to meet
    certain design constraints. For example, if signal integrity is of interest,
    the designer could use SWIDTH_DSPACE which spreads the wires farther apart
    from each other, or to reduce wire parasitics (i.e., RC), the designer can use
    DWIDTH_SSPACE.

 
    (c) PARM_buffering_scheme 

    PARM_buffering_scheme can be set to MIN_DELAY or STAGGERED.  
    MIN_DELAY buffering scheme uses the traditional buffering in which the number 
    and size of buffers are chosen such that the point-to-point delay is minimized.  
    The STAGGERED buffering scheme uses a circuit technique (cf. Kahng et al. DATE-1998) 
    to improve global interconnect performance. The idea is to reduce the worst-case 
    Miller coupling by offsetting the inverters on adjacent lines. 

 
    (d) PARM_shielding 

    PARM_shielding can be set to TRUE or FALSE. Shielding is the use of a grounded 
    wire between two wires of interest. Shielding will reduce crosstalk effects and 
    improve signal integrity.  However, it increases the area (i.e., adding another 
    wire) and also takes up a routing track.  The default value should be set to FALSE 
    to assess the worst-case impact on performance and power. 


(4.4) Clock Power Parameters  

    (a) PARM_pipeline_stages 
	matalab
    This is the number of pipeline stages in the router. 
    
    (b) PARM_H_tree_clock 

    PARM_H_tree_clock defines whether H_tree_clock power is included in the total
    clock power. If set to 1, router_diagonal value must be provided. 

    (c) PARM_router_diagonal 

    PARM_router_diagonal is the square root of the routing area in microns. This value 
    is used to calculate H_tree_clock power for the router. 
	
(4.5) ORION 3.0 Training Data  

    We supply a default set of training data points for 65nm and 45nm. 
	Data points for 65nm technology are stored in default_selected_area_65 and 
	default_selected_power_65, for area and power, respectively. 
	Data points for 45nm technology are stored in default_selected_area_45 and 
	default_selected_power_45, for area and power, respectively.
	
	In order to use your own training data set, use the -train option followed by the absolute path
	of the training data file. 
	
	The user-supplied training file must be formatted as follows: The area training files
	are formatted using	5 columns: B, V, P, F, and area value in micrometer-squared. The 
	power training files are formatted using 7 columns: B, V, P, F, internal power, 
	switching power, and leakage power. Power values should be supplied in mW.
	
(4.6) ORION 3.0 Testing Data

	ORION 3.0 reads SIM_port.h for B,V,P,F, and technology parameters by default. In order 
	to supply your own testing set, use the -test option followed by the absolute path of test data file.
	The file should be formatted using 4 columns of B,V,P,F combinations.
	
	A sample test file (test.txt) is provided for reference.

(5) MISC 
   
    This document provides what's new in ORION 3.0. For more general information about ORION, 
    please refer to README.ORION2.0  
 
    SIM_technology_v1.h file provides the technology parameters for 110nm and above. These were 
    copied from ORION 1.0 and thus remain the same. SIM_technology_v2.h and SIM_technology_v2_NEW.h 
	files contains the technology parameters we updated for 90nm, 65nm, 45nm and 32nm. Several 
	parameters in the SIM_technology_v2.h file were derived from Cacti 5.3 (Rnchannelon, 
	Rpchannelon, CamCellHeight, CamCellWidth, Rbitmetal, Rwordmetal). technology_area_power provides
	technology parameters used in ORION 3.0.

(6) Contacts  
    If you have any questions, please feel free to email us. Also, if you find bugs, please 
    let us know and we'll fix them as soon as possible. Thanks very much. 
	Andrew Kahng -- abk@cs.ucsd.edu
	Bill Lin -- billin@eng.ucsd.edu
	Siddartha Nath -- sinath@cs.ucsd.edu
	Jeremiah Fong -- jeremiahfong@gmail.com

*/